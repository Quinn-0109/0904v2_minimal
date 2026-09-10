#!/usr/bin/env python3

import math
import os
import sys
import unittest
from unittest import mock

import numpy as np

SCRIPTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, SCRIPTS)

from baseline_planning_core import (  # noqa: E402
    OccupancyGrid2D, astar_safe_path,
    preflight_entry_inside_anchor_replacement_allowed,
    portal_entry_inside_anchor_candidates,
    portal_entry_early_turn_candidates,
    portal_entry_live_corridor_anchor_skip_candidates,
    portal_entry_parallel_lane_candidates,
    portal_entry_aperture_lane_candidates,
    room_waypoint_force_heading, room_waypoint_heading,
    two_pose_edge_hint_priority_allowed,
)
from lightweight_room_core import (  # noqa: E402
    complete_reversed_entry_trace_to_corridor,
    direct_exit_fold_allowed,
    EstimatedDoorway, LightweightRoomConfig, LightweightRoomScheduler,
    obstacle_candidate_is_behind_front_gap,
    prepare_exit_path, plan_two_pose_room_sweep,
    stage_uncertain_open_near_mapping_view,
    select_mandatory_second_visual_view,
    select_opposite_side_second_visual_view, sparsify_verified_trace,
    select_truth_obstacle_gap_first_visual_view,
    select_front_gap_opposite_side_second_visual_view,
    select_locked_open_missing_depth_view,
    select_reversed_open_g3_near_view,
)


class LocalDoorPromotionTest(unittest.TestCase):
    def setUp(self):
        self.scheduler = LightweightRoomScheduler(
            LightweightRoomConfig(enabled=True))
        self.evidence = {
            "candidate_id": "local-door-1",
            "confirmed": True,
            "astar_reachable": True,
            "scan_lite_safe": True,
            "door_center": [2.0, 1.0],
            "corridor_side": [2.0, 0.4],
            "entry_goal": [2.0, 2.5],
            "yaw": 1.57079632679,
            "width_m": 1.2,
            "confirmation_count": 3,
        }

    def test_open_g4_recovery_does_not_backtrack_to_g3(self):
        door = EstimatedDoorway(
            door_id="estimated_open", center=(0.0, 0.0),
            normal_direction=0.0, width=1.4,
            left_frame_point=(0.0, -0.7), right_frame_point=(0.0, 0.7),
            corridor_side=(-0.6, 0.0), interior_side=(1.5, 0.0),
            confidence=0.9, entered_at=1.0,
            viewpoint_contract="open_deep_near")
        config = LightweightRoomConfig(
            enabled=True, minimum_goal_separation=1.8,
            semantic_completion_tolerance=0.8)
        proposal = select_reversed_open_g3_near_view(
            door, current=(5.3, 0.9), completed=[(5.5, 0.0)],
            executed_path=[(1.5, 0.0), (3.5, 0.0), (5.5, 0.0)],
            config=config)
        self.assertIsNotNone(proposal)
        self.assertEqual(proposal["two_pose_strategy"],
                         "direct_open_g4_recovery_to_near")
        self.assertFalse(proposal["verified_trajectory_backtrack"])
        self.assertTrue(proposal["live_3d_audit_required"])
        self.assertEqual(len(proposal["preflight_path"]), 2)
        self.assertEqual(tuple(proposal["preflight_path"][0]), (5.3, 0.9))
        self.assertLess(proposal["depth"], 3.0)

    def test_portal_entry_early_turn_stays_outside_then_keeps_all_anchors(self):
        current = (17.72, -0.20)
        anchors = [(19.7029, -0.2763), (19.71, -0.8762),
                   (19.7175, -1.5137)]
        candidates = portal_entry_early_turn_candidates(current, anchors)
        self.assertEqual(len(candidates), 3)
        for candidate in candidates:
            self.assertEqual(candidate[0], current)
            self.assertEqual(candidate[2:], anchors)
            bend = candidate[1]
            # The bend occurs before the door tangentially and farther out in
            # the corridor than the saved corridor-side anchor.
            self.assertLess(bend[0], anchors[0][0])
            self.assertGreater(bend[1], anchors[0][1])

    def test_portal_entry_early_turn_requires_lateral_approach(self):
        anchors = [(2.0, 0.4), (2.0, 1.0), (2.0, 2.5)]
        self.assertEqual(
            portal_entry_early_turn_candidates((2.0, -0.5), anchors), [])

    def test_live_corridor_anchor_skip_keeps_portal_and_inside(self):
        current = (0.10, 27.44)
        anchors = [(0.275, 28.503), (0.875, 28.492), (1.725, 28.477)]
        candidates = portal_entry_live_corridor_anchor_skip_candidates(
            current, anchors)
        self.assertGreaterEqual(len(candidates), 1)
        for candidate in candidates:
            self.assertEqual(candidate[0], current)
            self.assertEqual(candidate[-2:], anchors[1:])
            self.assertNotIn(anchors[0], candidate)

    def test_live_corridor_anchor_skip_rejects_already_inside(self):
        anchors = [(0.275, 28.503), (0.875, 28.492), (1.725, 28.477)]
        self.assertEqual(
            portal_entry_live_corridor_anchor_skip_candidates(
                (1.10, 28.49), anchors), [])

    def test_preflight_entry_allows_bounded_inside_anchor_shortening(self):
        anchors = [(20.165, -0.317), (20.179, 0.283), (20.175, 1.425)]
        refined = [(19.88, -0.07), anchors[0], anchors[1], (20.175, 0.975)]
        self.assertTrue(preflight_entry_inside_anchor_replacement_allowed(
            anchors, refined))

    def test_preflight_entry_rejects_missing_door_center(self):
        anchors = [(20.165, -0.317), (20.179, 0.283), (20.175, 1.425)]
        refined = [(19.88, -0.07), anchors[0], (20.175, 0.975)]
        self.assertFalse(preflight_entry_inside_anchor_replacement_allowed(
            anchors, refined))

    def test_preflight_entry_rejects_shallow_or_lateral_repair(self):
        anchors = [(20.165, -0.317), (20.179, 0.283), (20.175, 1.425)]
        shallow = [(19.88, -0.07), anchors[0], anchors[1], (20.177, 0.68)]
        lateral = [(19.88, -0.07), anchors[0], anchors[1], (20.50, 1.00)]
        self.assertFalse(preflight_entry_inside_anchor_replacement_allowed(
            anchors, shallow))
        self.assertFalse(preflight_entry_inside_anchor_replacement_allowed(
            anchors, lateral))

    def test_preflight_entry_allows_bounded_deeper_lateral_replacement(self):
        anchors = [(20.1944, -0.1498), (20.1903, 0.9319),
                   (20.1877, 1.6002)]
        replacement = [(20.1944, -0.1498), (20.1903, 0.9319),
                       (20.3375, 2.1318)]
        self.assertTrue(preflight_entry_inside_anchor_replacement_allowed(
            anchors, replacement))

    def test_inside_anchor_candidates_are_deeper_and_bounded(self):
        anchors = [(20.1944, -0.1498), (20.1903, 0.9319),
                   (20.1877, 1.6002)]
        candidates = portal_entry_inside_anchor_candidates(anchors)
        self.assertGreater(len(candidates), 0)
        original_depth = math.hypot(
            anchors[2][0] - anchors[1][0],
            anchors[2][1] - anchors[1][1])
        nx = (anchors[2][0] - anchors[1][0]) / original_depth
        ny = (anchors[2][1] - anchors[1][1]) / original_depth
        for point in candidates:
            dx, dy = point[0] - anchors[1][0], point[1] - anchors[1][1]
            depth = dx * nx + dy * ny
            lateral = abs(dx * ny - dy * nx)
            self.assertGreater(depth, original_depth + 0.05)
            self.assertLessEqual(depth, original_depth + 1.0 + 1e-6)
            self.assertLessEqual(lateral, 0.28 + 1e-6)

    def test_parallel_portal_lanes_shift_whole_ray_inside_door_width(self):
        anchors = [(-0.5736, 28.8953), (-1.1736, 28.8957),
                   (-2.0236, 28.8962)]
        candidates = portal_entry_parallel_lane_candidates(anchors, 1.4)
        self.assertEqual([round(item[0], 2) for item in candidates],
                         [0.18, -0.18, 0.28, -0.28])
        for offset, shifted in candidates:
            self.assertLessEqual(abs(offset), 0.30 + 1e-6)
            self.assertEqual(len(shifted), 3)
            # Uniform translation preserves both normal crossing legs.
            for original_a, original_b, shifted_a, shifted_b in zip(
                    anchors[:-1], anchors[1:], shifted[:-1], shifted[1:]):
                self.assertAlmostEqual(
                    shifted_b[0] - shifted_a[0],
                    original_b[0] - original_a[0], places=6)
                self.assertAlmostEqual(
                    shifted_b[1] - shifted_a[1],
                    original_b[1] - original_a[1], places=6)

    def test_parallel_portal_lanes_refuse_too_narrow_aperture(self):
        anchors = [(0.0, 0.0), (0.0, 0.6), (0.0, 1.4)]
        self.assertEqual(
            portal_entry_parallel_lane_candidates(anchors, 0.75), [])

    def test_aperture_lanes_keep_corridor_anchor_and_bound_extension(self):
        anchors = [(0.68, 28.89), (1.28, 28.89), (2.13, 28.89)]
        candidates = portal_entry_aperture_lane_candidates(anchors, 1.4)
        self.assertGreater(len(candidates), 4)
        for offset, extension, shifted in candidates:
            self.assertEqual(shifted[0], anchors[0])
            self.assertLessEqual(abs(offset), 0.30 + 1e-6)
            self.assertGreaterEqual(extension, 0.0)
            self.assertLessEqual(extension, 0.75 + 1e-6)
            self.assertAlmostEqual(
                math.hypot(shifted[2][0] - shifted[1][0],
                           shifted[2][1] - shifted[1][1]),
                0.85 + extension, places=6)

    def test_astar_goal_override_is_explicit_and_single_cell_only(self):
        data = np.zeros((7, 7), dtype=np.int16)
        data[3, 5] = 100
        grid = OccupancyGrid2D(data, 1.0, 0.0, 0.0)
        start, goal = (1.5, 3.5), (5.5, 3.5)
        self.assertEqual(
            astar_safe_path(grid, start, goal, 0.0, 0.1)["reason"],
            "goal_footprint_blocked")
        allowed = astar_safe_path(
            grid, start, goal, 0.0, 0.1, allow_blocked_goal=True)
        self.assertTrue(allowed["success"])
        self.assertEqual(allowed["reason"], "path_found_goal_override")
        data[3, 4] = 100
        detour = astar_safe_path(
            grid, start, goal, 0.0, 0.1, allow_blocked_goal=True)
        self.assertTrue(detour["success"])
        self.assertTrue(any(abs(point[1] - 3.5) > 0.5
                            for point in detour["path"][1:-1]))

    def test_compacted_single_waypoint_exit_still_forces_heading(self):
        endpoint = (0.58, 14.86)
        self.assertTrue(room_waypoint_force_heading(
            "EXIT", endpoint, endpoint))
        self.assertFalse(room_waypoint_force_heading(
            "ENTRY", endpoint, endpoint))

    def test_exit_l_shape_faces_current_leg_not_following_leg(self):
        heading = room_waypoint_heading(
            "EXIT", (0.0, 0.0), (1.0, 0.0), (1.0, 2.0))
        self.assertAlmostEqual(heading, 0.0, places=6)

    def test_exit_final_leg_faces_outward_endpoint(self):
        heading = room_waypoint_heading(
            "EXIT", (1.0, 0.0), (1.0, 2.0), (1.0, 2.0))
        self.assertAlmostEqual(heading, math.pi / 2.0, places=6)

    def test_non_exit_retains_lookahead_heading(self):
        heading = room_waypoint_heading(
            "G4", (0.0, 0.0), (1.0, 0.0), (1.0, 2.0))
        self.assertAlmostEqual(heading, math.pi / 2.0, places=6)

    def test_edge_hint_can_direct_first_two_pose_view(self):
        self.assertTrue(two_pose_edge_hint_priority_allowed(0, 26, 0, 0))

    def test_stale_edge_hint_cannot_duplicate_second_view_fan(self):
        self.assertFalse(two_pose_edge_hint_priority_allowed(1, 26, 0, 0))

    def test_confirmed_track_disables_edge_hint_priority(self):
        self.assertFalse(two_pose_edge_hint_priority_allowed(0, 2, 1, 3))

    def test_locomotion_recovery_does_not_consume_exit_retry(self):
        door = EstimatedDoorway(
            door_id="estimated_door_pause", center=(2.0, 1.0),
            normal_direction=1.57079632679, width=1.2,
            left_frame_point=(1.4, 1.0), right_frame_point=(2.6, 1.0),
            corridor_side=(2.0, 0.4), interior_side=(2.0, 2.5),
            confidence=0.9, entered_at=1.0)
        self.scheduler.active_door = door
        self.scheduler.active_room_id = "estimated_room_pause"
        self.scheduler.state = "ROOM_EXIT"
        self.scheduler.exit_attempts = 0
        result = self.scheduler.record_result(
            {"source": "lightweight_room_exit", "room_role": "EXIT",
             "room_id": "estimated_room_pause",
             "position": [2.0, 0.2, 0.0]},
            False, (2.0, 2.0), 12.0, "locomotion_not_ready")
        self.assertFalse(result["success"])
        self.assertTrue(result["transient_locomotion_failure"])
        self.assertEqual(self.scheduler.exit_attempts, 0)
        self.assertIs(self.scheduler.active_door, door)
        self.assertEqual(self.scheduler.state, "ROOM_EXIT")
        self.assertEqual(
            self.scheduler.events[-1]["event"],
            "ROOM_GOAL_DEFERRED_LOCOMOTION_RECOVERY")

    def test_confirmed_local_candidate_enters_common_preflight(self):
        sentinel = object()
        grid = OccupancyGrid2D(
            np.zeros((40, 40), dtype=np.int16), 0.15, 0.0, 0.0)
        self.scheduler._prepare_and_activate = mock.Mock(return_value=sentinel)
        result = self.scheduler.consider_local_door_candidate(
            grid, (2.0, 0.0), self.evidence, 12.0)
        self.assertIs(result, sentinel)
        door = self.scheduler._prepare_and_activate.call_args[0][1]
        self.assertEqual(door.center, (2.0, 1.0))
        self.assertEqual(door.corridor_side, (2.0, 0.4))
        self.assertEqual(door.interior_side, (2.0, 2.5))
        self.assertEqual(self.scheduler.events[-1]["event"],
                         "LOCAL_DOOR_CANDIDATE_PROMOTED")

    def test_unconfirmed_candidate_cannot_activate(self):
        self.evidence["confirmed"] = False
        self.scheduler._prepare_and_activate = mock.Mock()
        self.assertIsNone(self.scheduler.consider_local_door_candidate(
            object(), (2.0, 0.0), self.evidence, 12.0))
        self.scheduler._prepare_and_activate.assert_not_called()

    def test_failed_corridor_resume_can_be_released(self):
        self.scheduler.state = "CORRIDOR_RESUME"
        self.scheduler.abandon_corridor_resume(20.0, "failure_limit")
        self.assertEqual(self.scheduler.state, "CORRIDOR_SWEEP")

    def test_exit_preflight_bounds_each_local_astar(self):
        map_grid = OccupancyGrid2D(
            np.zeros((100, 100), dtype=np.int16), 0.15, 0.0, 0.0)
        config = LightweightRoomConfig(
            enabled=True, exit_preflight_maximum_expansions=321)
        door = EstimatedDoorway(
            door_id="estimated_door_01", center=(7.5, 7.5),
            normal_direction=0.0, width=1.20,
            left_frame_point=(7.5, 8.1),
            right_frame_point=(7.5, 6.9),
            corridor_side=(6.9, 7.5), interior_side=(9.0, 7.5),
            confidence=0.9, entered_at=1.0)
        with mock.patch(
                "lightweight_room_core.astar_safe_path",
                return_value={"success": False,
                              "reason": "maximum_expansions", "path": []}) as planner:
            self.assertIsNone(prepare_exit_path(
                map_grid, door, (8.5, 7.5), config))
        self.assertEqual(
            planner.call_args.kwargs["maximum_expansions"], 321)

    def test_unverified_initial_staging_uses_short_evidence_cooldown(self):
        scheduler = LightweightRoomScheduler(LightweightRoomConfig(
            enabled=True,
            require_portal_preflight_for_entry=True,
            door_evidence_retry_seconds=3.0,
            door_cooldown_seconds=90.0))
        door = EstimatedDoorway(
            door_id="estimated_door_01", center=(2.0, 1.0),
            normal_direction=1.57079632679, width=1.20,
            left_frame_point=(1.4, 1.0), right_frame_point=(2.6, 1.0),
            corridor_side=(2.0, 0.4), interior_side=(2.0, 2.5),
            confidence=0.9, entered_at=10.0)
        scheduler.active_door = door
        scheduler.active_room_id = "estimated_room_01"
        scheduler.room_started_at = 10.0
        scheduler.prepared_entry = {
            "position": [2.0, 1.25],
            "score": 0.0,
            "entry_staging_fallback": True,
        }
        grid = OccupancyGrid2D(
            np.zeros((40, 40), dtype=np.int16), 0.15, 0.0, 0.0)

        self.assertIsNone(scheduler.next_goal(grid, (2.0, 0.0), 12.0))
        cooldowns = scheduler.snapshot()["door_cooldowns"]
        self.assertEqual(len(cooldowns), 1)
        self.assertAlmostEqual(cooldowns[0]["until"], 15.0)
        self.assertIsNone(scheduler.active_door)

    def test_cooled_known_unvisited_door_can_reactivate_without_duplicate(self):
        door = EstimatedDoorway(
            door_id="estimated_door_03", center=(6.0, -1.0),
            normal_direction=-1.57079632679, width=1.0,
            left_frame_point=(6.5, -1.0), right_frame_point=(5.5, -1.0),
            corridor_side=(6.0, -0.4), interior_side=(6.0, -2.0),
            confidence=0.95, entered_at=20.0)
        self.scheduler.detector.doors.append(door)
        self.scheduler._set_door_cooldown(door.center, 40.0)
        sentinel = object()
        self.scheduler._prepare_and_activate = mock.Mock(
            return_value=sentinel)

        result = self.scheduler.retry_known_unvisited_door(
            object(), (6.0, 0.0), 41.0, 6.5)

        self.assertIs(result, sentinel)
        self.assertEqual(len(self.scheduler.detector.doors), 1)
        self.assertIs(
            self.scheduler._prepare_and_activate.call_args.args[1], door)
        self.assertTrue(
            self.scheduler._prepare_and_activate.call_args.kwargs[
                "reuse_existing"])
        self.assertEqual(
            self.scheduler.events[-1]["event"],
            "KNOWN_UNVISITED_DOOR_RETRY_SELECTED")

    def test_known_unvisited_door_waits_for_cooldown(self):
        door = EstimatedDoorway(
            door_id="estimated_door_03", center=(6.0, -1.0),
            normal_direction=-1.57079632679, width=1.0,
            left_frame_point=(6.5, -1.0), right_frame_point=(5.5, -1.0),
            corridor_side=(6.0, -0.4), interior_side=(6.0, -2.0),
            confidence=0.95, entered_at=20.0)
        self.scheduler.detector.doors.append(door)
        self.scheduler._set_door_cooldown(door.center, 40.0)
        self.scheduler._prepare_and_activate = mock.Mock()
        self.assertIsNone(self.scheduler.retry_known_unvisited_door(
            object(), (6.0, 0.0), 39.9, 6.5))
        self.scheduler._prepare_and_activate.assert_not_called()


    def test_open_room_relaxes_only_post_entry_path_after_normal_astar_fails(self):
        grid = OccupancyGrid2D(
            np.zeros((100, 100), dtype=np.int16), 0.15, 0.0, 0.0)
        config = LightweightRoomConfig(
            enabled=True, goal_clearance=0.38,
            post_entry_path_clearance=0.20,
            room_depth_probe_range=6.5, room_side_probe_range=5.0,
            maximum_side_lateral=4.25, visual_breadth_min_baseline=3.6,
            minimum_goal_separation=1.8)
        door = EstimatedDoorway(
            door_id="estimated_door_relaxed", center=(2.0, 7.5),
            normal_direction=0.0, width=1.20,
            left_frame_point=(2.0, 8.1),
            right_frame_point=(2.0, 6.9),
            corridor_side=(1.4, 7.5), interior_side=(3.4, 7.5),
            confidence=0.9, entered_at=1.0)

        def clearance_sensitive_path(_, start, target, clearance, *__args,
                                     **__kwargs):
            if clearance > 0.201:
                return {"success": False, "reason": "goal_unreachable",
                        "path": []}
            return {"success": True, "reason": "path_found",
                    "path": [start, target]}

        with mock.patch(
                "lightweight_room_core._observed_clear_straight_segment",
                return_value=False), mock.patch(
                "lightweight_room_core.astar_safe_path",
                side_effect=clearance_sensitive_path):
            route = plan_two_pose_room_sweep(
                grid, door, (3.4, 7.5), door.interior_side, config)

        self.assertIsNotNone(route)
        self.assertEqual(len(route["ordered"]), 2)
        self.assertTrue(all(
            item["post_entry_path_clearance_relaxed"]
            for item in route["ordered"]))
        self.assertTrue(all(
            abs(item["path_planning_clearance_m"] - 0.20) < 1e-6
            for item in route["ordered"]))

    def test_two_pose_clear_room_visits_deep_then_axial_near(self):
        grid = OccupancyGrid2D(
            np.zeros((160, 160), dtype=np.int16), 0.10, -8.0, -8.0)
        config = LightweightRoomConfig(
            enabled=True, room_depth_probe_range=6.5,
            maximum_side_lateral=5.0, minimum_goal_separation=1.8,
            open_room_preferred_deep_depth=4.75,
            open_room_minimum_deep_depth=4.50)
        door = EstimatedDoorway(
            door_id="estimated_door_01", center=(0.0, 0.0),
            normal_direction=0.0, width=1.2,
            left_frame_point=(0.0, 0.6), right_frame_point=(0.0, -0.6),
            corridor_side=(-0.6, 0.0), interior_side=(1.1, 0.0),
            confidence=0.9, entered_at=1.0)

        route = plan_two_pose_room_sweep(
            grid, door, (1.1, 0.0), (1.1, 0.0), config)

        self.assertIsNotNone(route)
        self.assertEqual(route["strategy"], "deep_then_axial_near")
        self.assertEqual(len(route["ordered"]), 2)
        first, second = route["ordered"]
        self.assertGreater(first["depth"], second["depth"])
        self.assertGreaterEqual(first["depth"], 4.45)
        self.assertLess(first["depth"], 5.05)
        self.assertGreaterEqual(first["depth"] - second["depth"], 2.0)
        self.assertLess(second["depth"], 3.0)
        self.assertGreater(route["baseline_m"], 2.3)
        # Open rooms use the short doorway-normal return lane. Large lateral
        # triangles are reserved for a coherent door-front obstacle.
        self.assertLessEqual(abs(first["lateral"]), 0.8)
        self.assertLessEqual(abs(second["lateral"]), 0.8)

    def test_same_side_g4_is_repaired_to_fresh_opposite_side(self):
        grid = OccupancyGrid2D(
            np.zeros((160, 160), dtype=np.int16), 0.10, -8.0, -8.0)
        door = EstimatedDoorway(
            door_id="estimated_door_repair", center=(0.0, 0.0),
            normal_direction=0.0, width=1.2,
            left_frame_point=(0.0, 0.6), right_frame_point=(0.0, -0.6),
            corridor_side=(-0.6, 0.0), interior_side=(1.1, 0.0),
            confidence=0.9, entered_at=1.0)
        config = LightweightRoomConfig(
            enabled=True, visual_breadth_min_baseline=3.6,
            minimum_goal_separation=1.8)
        anchor = (4.8, -1.6)
        template = {
            "position": (2.7, -0.9), "role": "G4",
            "coverage_cells": set(), "adaptive_marginal_gain_m2": 0.2,
            "adaptive_utility": 0.0, "preflight_path_length_m": 1.0}

        repaired = select_opposite_side_second_visual_view(
            grid, door, anchor, [anchor], template, config)

        self.assertIsNotNone(repaired)
        self.assertGreater(door.lateral(repaired["position"]), 0.6)
        self.assertGreaterEqual(repaired["visual_baseline_m"], 2.4)
        self.assertEqual(repaired["two_pose_strategy"],
                         "fresh_grid_opposite_side_repair")

    def test_mandatory_second_view_falls_back_to_safe_deep_point(self):
        grid = OccupancyGrid2D(
            np.zeros((120, 120), dtype=np.int16), 0.15, 0.0, 0.0)
        door = EstimatedDoorway(
            door_id="estimated_door_upper", center=(9.0, 3.0),
            normal_direction=1.57079632679, width=0.75,
            left_frame_point=(8.625, 3.0),
            right_frame_point=(9.375, 3.0),
            corridor_side=(9.0, 2.4), interior_side=(9.0, 4.0),
            confidence=0.95, entered_at=1.0)
        config = LightweightRoomConfig(
            enabled=True, minimum_goal_separation=1.8,
            open_room_preferred_deep_depth=4.75,
            visual_two_pose_lateral_enabled=True)
        first = (9.0, 5.35)
        second = select_mandatory_second_visual_view(
            grid, door, first, [first], config)
        self.assertIsNotNone(second)
        self.assertEqual(second["role"], "G4")
        self.assertEqual(second["two_pose_index"], 2)
        self.assertGreaterEqual(second["visual_baseline_m"], 1.8)
        self.assertGreater(door.depth(second["position"]), door.depth(first))
        self.assertGreater(second["preflight_path_length_m"], 0.0)

    def test_deep_first_view_gets_diagonal_near_partner(self):
        grid = OccupancyGrid2D(
            np.zeros((160, 160), dtype=np.int16), 0.10, -8.0, -8.0)
        door = EstimatedDoorway(
            door_id="estimated_door_deep_first", center=(0.0, 0.0),
            normal_direction=0.0, width=1.2,
            left_frame_point=(0.0, 0.6), right_frame_point=(0.0, -0.6),
            corridor_side=(-0.6, 0.0), interior_side=(1.1, 0.0),
            confidence=0.9, entered_at=1.0)
        config = LightweightRoomConfig(
            enabled=True, minimum_goal_separation=1.8,
            open_room_preferred_deep_depth=4.75,
            visual_two_pose_lateral_enabled=True)
        first = (4.0, 1.35)

        second = select_mandatory_second_visual_view(
            grid, door, first, [first], config)

        self.assertIsNotNone(second)
        depth_delta = abs(door.depth(first) -
                          door.depth(second["position"]))
        lateral_delta = abs(door.lateral(first) -
                            door.lateral(second["position"]))
        self.assertGreaterEqual(depth_delta, 1.8)
        self.assertLessEqual(abs(second["lateral"]), 0.8)
        self.assertGreaterEqual(second["visual_baseline_m"], 1.8)
        self.assertEqual(second["two_pose_strategy"],
                         "mandatory_deep_then_axial_near_fallback")

    def test_open_g3_completion_forces_g4_replan_from_actual_pose(self):
        source_path = os.path.join(SCRIPTS, "lightweight_room_core.py")
        with open(source_path, encoding="utf-8") as stream:
            source = stream.read()
        self.assertIn("ROOM_TWO_POSE_OPEN_G4_FRESH_REPLAN", source)
        branch = source.split(
            "ROOM_TWO_POSE_OPEN_G4_FRESH_REPLAN", 1)[0][-2200:]
        self.assertIn('"deep_then_axial_near"', branch)
        self.assertIn("self.adaptive_route_queue = []", branch)
        self.assertIn("actual completed G3 pose", branch)

    def test_obstacle_g4_mapping_transit_reuses_executed_g3_trace(self):
        source_path = os.path.join(SCRIPTS, "lightweight_room_core.py")
        with open(source_path, encoding="utf-8") as stream:
            source = stream.read()
        save_branch = source.split(
            "ROOM_OBSTACLE_G3_EXECUTED_ROUTE_SAVED", 1)[0][-1800:]
        self.assertIn('diagnostic.get("preflight_path")', save_branch)
        self.assertIn("obstacle_front_gap_first", save_branch)
        transit_branch = source.split(
            "reverse_physically_executed_g3_trace", 1)[0][-1800:]
        self.assertIn("reversed(", transit_branch)
        self.assertIn("self.obstacle_gap_g3_executed_path", transit_branch)
        self.assertIn("transit_path", transit_branch)

    def test_locked_two_view_debt_overrides_single_view_adaptive_cap(self):
        source_path = os.path.join(SCRIPTS, "lightweight_room_core.py")
        with open(source_path, encoding="utf-8") as stream:
            source = stream.read()
        # The timed upper-floor profile uses a one-view adaptive cap, but a
        # locked open/obstacle contract still owes a real separated G4 after
        # G3.  Both the capacity calculation and proposal-generation gate
        # must preserve that mandatory slot.
        capacity = source.split("observation_capacity = (", 1)[1][:450]
        self.assertIn("mandatory_two_pose_pending", capacity)
        proposal_gate = source.split(
            "len(attempted) <\n                     "
            "self.config.adaptive_maximum_viewpoints", 1)[1][:180]
        self.assertIn("mandatory_two_pose_pending", proposal_gate)

    def test_short_portal_disconnect_override_keeps_dense_3d_audit(self):
        manager_path = os.path.join(
            SCRIPTS, "baseline_exploration_manager.py")
        with open(manager_path, encoding="utf-8") as stream:
            source = stream.read()
        branch = source.split(
            "ROOM_ENTRY_2D_DISCONNECT_LIVE_3D_OVERRIDE", 1)[0][-5200:]
        self.assertIn('segment.get("reason") == "goal_unreachable"', branch)
        self.assertIn("segment_distance <= 1.25", branch)
        self.assertIn("segment_distance / 0.10", branch)
        self.assertIn("not audit.map_available", branch)
        self.assertIn("audit.occupied_collision", branch)

    def test_two_pose_central_obstacle_splits_views_across_both_sides(self):
        data = np.zeros((160, 160), dtype=np.int16)
        resolution, origin = 0.10, -8.0
        for wx in np.arange(2.35, 3.16, resolution):
            for wy in np.arange(-0.55, 0.56, resolution):
                x = int((wx - origin) / resolution)
                y = int((wy - origin) / resolution)
                data[y, x] = 100
        grid = OccupancyGrid2D(data, resolution, origin, origin)
        config = LightweightRoomConfig(
            enabled=True, room_depth_probe_range=6.5,
            maximum_side_lateral=5.0, minimum_goal_separation=1.8)
        door = EstimatedDoorway(
            door_id="estimated_door_01", center=(0.0, 0.0),
            normal_direction=0.0, width=1.2,
            left_frame_point=(0.0, 0.6), right_frame_point=(0.0, -0.6),
            corridor_side=(-0.6, 0.0), interior_side=(1.1, 0.0),
            confidence=0.9, entered_at=1.0)

        route = plan_two_pose_room_sweep(
            grid, door, (1.1, 0.0), (1.1, 0.0), config)

        self.assertIsNotNone(route)
        self.assertEqual(route["strategy"], "obstacle_side_split")
        first, second = route["ordered"]
        self.assertLess(first["lateral"] * second["lateral"], 0.0)
        self.assertGreater(route["baseline_m"], 1.8)
        self.assertGreaterEqual(min(first["depth"], second["depth"]),
                                2.30)
        self.assertGreater(route["baseline_m"], 1.8)

    def test_two_pose_central_obstacle_finishes_near_door_when_enabled(self):
        data = np.zeros((160, 160), dtype=np.int16)
        resolution, origin = 0.10, -8.0
        for wx in np.arange(2.35, 3.16, resolution):
            for wy in np.arange(-0.55, 0.56, resolution):
                x = int((wx - origin) / resolution)
                y = int((wy - origin) / resolution)
                data[y, x] = 100
        grid = OccupancyGrid2D(data, resolution, origin, origin)
        config = LightweightRoomConfig(
            enabled=True, room_depth_probe_range=6.5,
            maximum_side_lateral=5.0, minimum_goal_separation=1.8,
            obstacle_second_view_near_door=True)
        door = EstimatedDoorway(
            door_id="estimated_door_03", center=(0.0, 0.0),
            normal_direction=0.0, width=1.2,
            left_frame_point=(0.0, 0.6), right_frame_point=(0.0, -0.6),
            corridor_side=(-0.6, 0.0), interior_side=(1.1, 0.0),
            confidence=0.9, entered_at=1.0)

        route = plan_two_pose_room_sweep(
            grid, door, (1.1, 0.0), (1.1, 0.0), config)

        self.assertIsNotNone(route)
        self.assertEqual(
            route["strategy"], "obstacle_deep_then_opposite_near")
        first, second = route["ordered"]
        self.assertLess(first["lateral"] * second["lateral"], 0.0)
        self.assertGreaterEqual(first["depth"] - second["depth"], 1.0)
        self.assertLess(second["depth"], 2.35)
        self.assertTrue(route["obstacle"]["second_view_near_door"])
        self.assertLess(route["return_distance_m"], 3.5)

    def test_two_pose_central_obstacle_uses_front_opposite_sides_when_enabled(self):
        data = np.zeros((160, 160), dtype=np.int16)
        resolution, origin = 0.10, -8.0
        for wx in np.arange(1.70, 2.51, resolution):
            for wy in np.arange(-0.55, 0.56, resolution):
                x = int((wx - origin) / resolution)
                y = int((wy - origin) / resolution)
                data[y, x] = 100
        grid = OccupancyGrid2D(data, resolution, origin, origin)
        config = LightweightRoomConfig(
            enabled=True, room_depth_probe_range=6.5,
            maximum_side_lateral=5.0, minimum_goal_separation=1.8,
            obstacle_second_view_near_door=True,
            obstacle_front_opposite_side_pair=True)
        door = EstimatedDoorway(
            door_id="estimated_door_front_pair", center=(0.0, 0.0),
            normal_direction=0.0, width=1.2,
            left_frame_point=(0.0, 0.6), right_frame_point=(0.0, -0.6),
            corridor_side=(-0.6, 0.0), interior_side=(1.1, 0.0),
            confidence=0.9, entered_at=1.0)

        route = plan_two_pose_room_sweep(
            grid, door, (1.5, 0.0), (1.1, 0.0), config)

        self.assertIsNotNone(route)
        self.assertEqual(route["strategy"], "obstacle_front_side_split")
        first, second = route["ordered"]
        self.assertLess(first["lateral"] * second["lateral"], 0.0)
        self.assertLess(max(first["depth"], second["depth"]), 1.70)
        self.assertLessEqual(
            max(first["depth"], second["depth"]),
            route["obstacle"]["front_edge_depth_m"] - config.goal_clearance)
        self.assertAlmostEqual(first["depth"], second["depth"], delta=0.35)
        self.assertGreaterEqual(route["baseline_m"], 1.75)
        # The camera centres straddle the furniture but stay close enough to
        # cross the door--obstacle gap directly instead of hugging both room
        # walls and forcing a long wrap-around route.
        obstacle = route["obstacle"]
        self.assertLessEqual(
            max(abs(first["lateral"]), abs(second["lateral"])),
            max(abs(obstacle["minimum_lateral_m"]),
                abs(obstacle["maximum_lateral_m"])) +
            config.goal_clearance + 0.30)

    def test_far_central_object_keeps_open_room_deep_near_pair(self):
        data = np.zeros((160, 160), dtype=np.int16)
        resolution, origin = 0.10, -8.0
        for wx in np.arange(4.25, 5.26, resolution):
            for wy in np.arange(-0.55, 0.56, resolution):
                data[int((wy - origin) / resolution),
                     int((wx - origin) / resolution)] = 100
        grid = OccupancyGrid2D(data, resolution, origin, origin)
        config = LightweightRoomConfig(
            enabled=True, room_depth_probe_range=6.5,
            maximum_side_lateral=5.0, minimum_goal_separation=1.8,
            visual_breadth_min_baseline=3.6,
            open_room_preferred_deep_depth=4.75,
            open_room_minimum_deep_depth=4.50,
            obstacle_second_view_near_door=True,
            obstacle_front_opposite_side_pair=True)
        door = EstimatedDoorway(
            door_id="estimated_door_far_object", center=(0.0, 0.0),
            normal_direction=0.0, width=1.2,
            left_frame_point=(0.0, 0.6), right_frame_point=(0.0, -0.6),
            corridor_side=(-0.6, 0.0), interior_side=(1.1, 0.0),
            confidence=0.9, entered_at=1.0)

        route = plan_two_pose_room_sweep(
            grid, door, (1.5, 0.0), (1.1, 0.0), config)

        self.assertIsNotNone(route)
        self.assertEqual(route["strategy"], "deep_then_axial_near")
        first, second = route["ordered"]
        self.assertGreaterEqual(first["depth"], 4.45)
        self.assertGreater(first["depth"], second["depth"])
        self.assertLessEqual(abs(first["lateral"]), 1.2)
        self.assertLessEqual(abs(second["lateral"]), 0.8)

    def test_truth_open_contract_rejects_stale_central_smear(self):
        data = np.zeros((160, 160), dtype=np.int16)
        resolution, origin = 0.10, -8.0
        # A sizeable connected projection in the doorway-normal band would
        # ordinarily select the obstacle strategy.  On an upper floor this
        # can be a stale LIO smear spanning unrelated furniture; the physical
        # layout contract says this room is open.
        for wx in np.arange(3.00, 3.61, resolution):
            for wy in np.arange(-0.65, 0.66, resolution):
                data[int((wy - origin) / resolution),
                     int((wx - origin) / resolution)] = 100
        grid = OccupancyGrid2D(data, resolution, origin, origin)
        config = LightweightRoomConfig(
            enabled=True, room_depth_probe_range=7.2,
            maximum_side_lateral=5.0, minimum_goal_separation=1.8,
            visual_breadth_min_baseline=3.6,
            open_room_preferred_deep_depth=6.70,
            open_room_minimum_deep_depth=6.25,
            obstacle_front_opposite_side_pair=True)
        door = EstimatedDoorway(
            door_id="estimated_f2_room2", center=(0.0, 0.0),
            normal_direction=0.0, width=1.2,
            left_frame_point=(0.0, 0.6), right_frame_point=(0.0, -0.6),
            corridor_side=(-0.6, 0.0), interior_side=(1.1, 0.0),
            confidence=0.9, entered_at=1.0,
            viewpoint_contract="open_deep_near",
            viewpoint_contract_source="generated_floor_layout_physical_geometry",
            truth_room_id="floor_1_room_2")

        route = plan_two_pose_room_sweep(
            grid, door, (1.5, 0.0), (1.1, 0.0), config)

        self.assertIsNotNone(route)
        self.assertEqual(route["strategy"], "deep_then_axial_near")
        first, second = route["ordered"]
        self.assertGreaterEqual(first["depth"], 6.20)
        self.assertGreaterEqual(first["depth"] - second["depth"], 2.0)
        self.assertLess(second["depth"], 3.0)

    def test_truth_open_sparse_upper_map_seeds_centerline_deep_near_pair(self):
        data = np.full((100, 140), -1, dtype=np.int16)
        resolution, origin_x, origin_y = 0.10, -2.0, -5.0
        # Only the entry neighbourhood is currently observed.  This is the
        # F3 post-reset condition that used to create an off-axis provisional
        # pair and a long A* wrap through stale/unknown cells.
        for wx in np.arange(0.8, 2.61, resolution):
            for wy in np.arange(-0.35, 0.36, resolution):
                data[int((wy - origin_y) / resolution),
                     int((wx - origin_x) / resolution)] = 0
        grid = OccupancyGrid2D(
            data, resolution, origin_x, origin_y)
        config = LightweightRoomConfig(
            enabled=True, room_depth_probe_range=7.2,
            minimum_goal_separation=1.8,
            visual_breadth_min_baseline=3.6,
            open_room_preferred_deep_depth=6.7,
            open_room_minimum_deep_depth=6.25)
        door = EstimatedDoorway(
            door_id="f3-open-sparse", center=(0.0, 0.0),
            normal_direction=0.0, width=1.2,
            left_frame_point=(0.0, 0.6), right_frame_point=(0.0, -0.6),
            corridor_side=(-0.6, 0.0), interior_side=(1.1, 0.0),
            confidence=0.9, entered_at=1.0,
            viewpoint_contract="open_deep_near",
            viewpoint_contract_source=
                "generated_floor_layout_physical_geometry",
            truth_open_centerline_deep_depth_m=6.7)

        route = plan_two_pose_room_sweep(
            grid, door, (1.8, 0.0), (1.1, 0.0), config)

        self.assertIsNotNone(route)
        self.assertEqual(
            route["strategy"],
            "truth_layout_open_centerline_deep_near")
        deep, near = route["ordered"]
        self.assertEqual((deep["role"], near["role"]), ("G3", "G4"))
        self.assertGreaterEqual(deep["depth"], 6.25)
        # The near pose remains close to the door while respecting the
        # scheduler's rule that a post-ENTRY observation must advance at
        # least 0.55 m beyond the current body centre.
        self.assertLessEqual(near["depth"], 3.00)
        self.assertAlmostEqual(deep["lateral"], 0.0)
        self.assertAlmostEqual(near["lateral"], 0.0)
        self.assertTrue(deep["live_3d_audit_required"])
        self.assertGreaterEqual(route["baseline_m"], 3.6)

    def test_truth_open_retry_excludes_unreachable_deep_lane(self):
        grid = OccupancyGrid2D(
            np.zeros((180, 180), dtype=np.int16), 0.10, -5.0, -9.0)
        config = LightweightRoomConfig(
            enabled=True, room_depth_probe_range=7.2,
            minimum_goal_separation=1.8,
            visual_breadth_min_baseline=3.6,
            open_room_preferred_deep_depth=6.7,
            open_room_minimum_deep_depth=6.25)
        door = EstimatedDoorway(
            door_id="open-unreachable-lane", center=(0.0, 0.0),
            normal_direction=0.0, width=1.2,
            left_frame_point=(0.0, 0.6), right_frame_point=(0.0, -0.6),
            corridor_side=(-0.6, 0.0), interior_side=(1.1, 0.0),
            confidence=0.9, entered_at=1.0,
            viewpoint_contract="open_deep_near",
            truth_open_centerline_deep_depth_m=6.7)
        first = plan_two_pose_room_sweep(
            grid, door, (1.8, 0.0), (1.1, 0.0), config)
        rejected = first["ordered"][0]["position"]

        retry = plan_two_pose_room_sweep(
            grid, door, (1.8, 0.0), (1.1, 0.0), config,
            excluded_targets_by_role={"G3": [rejected]})

        self.assertIsNotNone(retry)
        self.assertGreaterEqual(
            abs(retry["ordered"][0]["position"][1] - rejected[1]), 0.30)
        self.assertGreaterEqual(retry["ordered"][0]["depth"], 6.25)

    def test_truth_open_contract_overrides_provisional_off_axis_pair(self):
        grid = OccupancyGrid2D(
            np.zeros((180, 180), dtype=np.int16), 0.10, -5.0, -9.0)
        config = LightweightRoomConfig(
            enabled=True, room_depth_probe_range=7.2,
            minimum_goal_separation=1.8,
            visual_breadth_min_baseline=3.6,
            open_room_preferred_deep_depth=6.7,
            open_room_minimum_deep_depth=6.25)
        door = EstimatedDoorway(
            door_id="f3-open-truth-prior", center=(0.0, 0.0),
            normal_direction=0.0, width=1.2,
            left_frame_point=(0.0, 0.6), right_frame_point=(0.0, -0.6),
            corridor_side=(-0.6, 0.0), interior_side=(1.1, 0.0),
            confidence=0.9, entered_at=1.0,
            viewpoint_contract="open_deep_near",
            viewpoint_contract_source=
                "generated_floor_layout_physical_geometry",
            truth_open_centerline_deep_depth_m=6.7)

        route = plan_two_pose_room_sweep(
            grid, door, (1.8, 0.55), (1.1, 0.0), config)

        self.assertEqual(
            route["strategy"], "truth_layout_open_centerline_deep_near")
        deep, near = route["ordered"]
        self.assertEqual((deep["lateral"], near["lateral"]), (0.0, 0.0))
        self.assertEqual((deep["role"], near["role"]), ("G3", "G4"))

    def test_truth_open_blocked_centerline_selects_parallel_axial_lane(self):
        data = np.zeros((180, 180), dtype=np.int16)
        resolution, origin_x, origin_y = 0.10, -3.0, -9.0
        # Paint only the exact deep centre endpoint as occupied.  The bounded
        # parallel lanes remain free and must keep the same lateral offset at
        # both deep and near views.
        deep_x, deep_y = 6.7, 0.0
        row = int((deep_y - origin_y) / resolution)
        col = int((deep_x - origin_x) / resolution)
        data[row - 2:row + 3, col - 2:col + 3] = 100
        grid = OccupancyGrid2D(data, resolution, origin_x, origin_y)
        config = LightweightRoomConfig(
            enabled=True, room_depth_probe_range=7.2,
            minimum_goal_separation=1.8,
            visual_breadth_min_baseline=3.6,
            open_room_preferred_deep_depth=6.7,
            open_room_minimum_deep_depth=6.25)
        door = EstimatedDoorway(
            door_id="open-parallel-lane", center=(0.0, 0.0),
            normal_direction=0.0, width=1.2,
            left_frame_point=(0.0, 0.6), right_frame_point=(0.0, -0.6),
            corridor_side=(-0.6, 0.0), interior_side=(1.1, 0.0),
            confidence=0.9, entered_at=1.0,
            viewpoint_contract="open_deep_near",
            truth_open_centerline_deep_depth_m=6.7)

        route = plan_two_pose_room_sweep(
            grid, door, (1.8, 0.0), (1.1, 0.0), config)

        deep, near = route["ordered"]
        self.assertNotAlmostEqual(deep["lateral"], 0.0)
        self.assertAlmostEqual(deep["lateral"], near["lateral"])
        self.assertEqual((deep["role"], near["role"]), ("G3", "G4"))
        self.assertGreater(deep["depth"] - near["depth"], 3.5)

    def test_fresh_f3_truth_open_pair_maps_near_before_deep(self):
        data = np.full((100, 140), -1, dtype=np.int16)
        resolution, origin_x, origin_y = 0.10, -2.0, -5.0
        for wx in np.arange(0.8, 2.61, resolution):
            for wy in np.arange(-0.35, 0.36, resolution):
                data[int((wy - origin_y) / resolution),
                     int((wx - origin_x) / resolution)] = 0
        grid = OccupancyGrid2D(data, resolution, origin_x, origin_y)
        config = LightweightRoomConfig(
            enabled=True, room_depth_probe_range=7.2,
            minimum_goal_separation=1.8,
            visual_breadth_min_baseline=3.6,
            open_room_preferred_deep_depth=6.7,
            open_room_minimum_deep_depth=6.25,
            truth_open_near_first=True)
        door = EstimatedDoorway(
            door_id="f3-open-near-first", center=(0.0, 0.0),
            normal_direction=0.0, width=1.2,
            left_frame_point=(0.0, 0.6), right_frame_point=(0.0, -0.6),
            corridor_side=(-0.6, 0.0), interior_side=(1.1, 0.0),
            confidence=0.9, entered_at=1.0,
            viewpoint_contract="open_deep_near",
            viewpoint_contract_source=
                "generated_floor_layout_physical_geometry",
            truth_open_centerline_deep_depth_m=6.7)

        route = plan_two_pose_room_sweep(
            grid, door, (1.8, 0.0), (1.1, 0.0), config)

        near, deep = route["ordered"]
        self.assertEqual((near["role"], deep["role"]), ("G3", "G4"))
        self.assertLessEqual(near["depth"], 3.0)
        self.assertGreaterEqual(deep["depth"], 6.25)
        self.assertTrue(route["truth_layout_near_first_mapping"])
        self.assertGreaterEqual(route["baseline_m"], 3.6)

    def test_locked_open_near_mapping_view_requires_fresh_deep_partner(self):
        grid = OccupancyGrid2D(
            np.zeros((180, 180), dtype=np.int16), 0.10, -9.0, -9.0)
        config = LightweightRoomConfig(
            enabled=True, goal_clearance=0.38,
            room_depth_probe_range=7.2,
            minimum_goal_separation=1.8,
            open_room_preferred_deep_depth=6.25,
            open_room_minimum_deep_depth=6.0,
            semantic_completion_tolerance=0.8)
        door = EstimatedDoorway(
            door_id="locked-open", center=(0.0, 0.0),
            normal_direction=0.0, width=1.2,
            left_frame_point=(0.0, 0.6), right_frame_point=(0.0, -0.6),
            corridor_side=(-0.6, 0.0), interior_side=(1.1, 0.0),
            confidence=0.9, entered_at=1.0,
            viewpoint_contract="open_deep_near")

        proposal = select_locked_open_missing_depth_view(
            grid, door, (2.1, 0.0), [(2.0, 0.1)], config)

        self.assertIsNotNone(proposal)
        self.assertEqual(proposal["two_pose_strategy"],
                         "locked_open_missing_deep")
        self.assertGreaterEqual(proposal["depth"], 5.55)
        self.assertGreaterEqual(proposal["visual_baseline_m"], 1.8)

    def test_locked_open_executed_deep_pose_requires_near_partner(self):
        """Planner and final contract must classify tolerated G3 equally."""
        grid = OccupancyGrid2D(
            np.zeros((180, 180), dtype=np.int16), 0.10, -9.0, -9.0)
        config = LightweightRoomConfig(
            enabled=True, goal_clearance=0.38,
            room_depth_probe_range=7.2,
            minimum_goal_separation=1.8,
            open_room_preferred_deep_depth=6.25,
            open_room_minimum_deep_depth=6.25,
            semantic_completion_tolerance=0.8)
        door = EstimatedDoorway(
            door_id="locked-open-tolerated-deep", center=(0.0, 0.0),
            normal_direction=0.0, width=1.2,
            left_frame_point=(0.0, 0.6), right_frame_point=(0.0, -0.6),
            corridor_side=(-0.6, 0.0), interior_side=(1.1, 0.0),
            confidence=0.9, entered_at=1.0,
            viewpoint_contract="open_deep_near")

        proposal = select_locked_open_missing_depth_view(
            grid, door, (5.48, 0.0), [(5.48, 0.0)], config)

        self.assertIsNotNone(proposal)
        self.assertEqual(proposal["two_pose_strategy"],
                         "locked_open_missing_near")
        self.assertLessEqual(proposal["depth"], 3.98)
        self.assertGreaterEqual(proposal["visual_baseline_m"], 1.8)

    def test_locked_open_missing_deep_uses_one_bounded_map_refresh(self):
        """A stale post-G3 grid must not immediately leave a partial room."""
        grid = OccupancyGrid2D(
            np.zeros((60, 60), dtype=np.int16), 0.10, -1.0, -3.0)
        config = LightweightRoomConfig(
            enabled=True, goal_clearance=0.38,
            room_depth_probe_range=7.2,
            minimum_goal_separation=1.8,
            open_room_preferred_deep_depth=6.25,
            open_room_minimum_deep_depth=6.0,
            semantic_completion_tolerance=0.8,
            visual_two_pose_lateral_enabled=True,
            adaptive_minimal_viewpoints=True)
        scheduler = LightweightRoomScheduler(config)
        door = EstimatedDoorway(
            door_id="locked-open", center=(0.0, 0.0),
            normal_direction=0.0, width=1.2,
            left_frame_point=(0.0, 0.6), right_frame_point=(0.0, -0.6),
            corridor_side=(-0.6, 0.0), interior_side=(1.1, 0.0),
            confidence=0.9, entered_at=1.0,
            room_id="estimated_room_01",
            viewpoint_contract="open_deep_near")
        scheduler.active_door = door
        scheduler.active_room_id = door.room_id
        scheduler.entry_confirmed = True
        scheduler.room_started_at = 1.0
        scheduler.return_anchor = door.interior_side
        scheduler.state = "G3_LEFT"
        scheduler.visual_deepening_requested = True
        scheduler.completed_points = [(3.6, 0.0)]
        scheduler.successful_roles = {"G3"}
        scheduler.role_attempts = {"G3": 1}
        scheduler.adaptive_route_queue = [{
            "position": (1.9, 0.0), "role": "G4", "score": 0.0,
            "clearance": 1.0, "coverage_cells": set(),
            "adaptive_marginal_gain_m2": 0.0,
            "adaptive_utility": 0.0,
            "preflight_path_length_m": 1.7,
            "preflight_path": [(3.6, 0.0), (1.9, 0.0)],
            "two_pose_room_sweep": True,
            "two_pose_strategy": "provisional_obstacle_side_split",
            "two_pose_index": 2,
        }]

        goal = scheduler.next_goal(grid, (3.6, 0.0), 5.0)

        self.assertIsNotNone(goal)
        self.assertEqual(goal["room_role"], "G4")
        self.assertEqual(goal["position"][:2], [3.6, 0.0])
        diagnostic = goal["room_goal_diagnostic"]
        self.assertTrue(diagnostic["entry_pose_visual_fallback"])
        self.assertEqual(
            diagnostic["two_pose_strategy"],
            "locked_open_missing_depth_map_refresh")
        self.assertEqual(scheduler.open_missing_depth_refresh_used, 1)
        self.assertIn(
            "ROOM_LOCKED_OPEN_MISSING_DEPTH_MAP_REFRESH",
            [event["event"] for event in scheduler.events])

        result = scheduler.record_result(
            goal, True, (3.6, 0.0), 6.0, "already_at_path_end")

        self.assertTrue(result["success"])
        self.assertEqual(scheduler.completed_points, [(3.6, 0.0)])
        self.assertEqual(scheduler.successful_roles, {"G3"})
        self.assertNotIn("G4", scheduler.role_attempts)
        self.assertNotIn(
            "duplicate_observation_pose_rejected",
            [event.get("reason") for event in scheduler.events])
        self.assertIn(
            "ROOM_ENTRY_MAP_REFRESH_COMPLETED",
            [event["event"] for event in scheduler.events])

    def test_failed_open_axial_g3_near_safe_target_is_committed(self):
        """A timed-out deep axial view is still a real physical viewpoint."""
        config = LightweightRoomConfig(
            enabled=True, minimum_goal_separation=1.8,
            open_room_minimum_deep_depth=6.25,
            open_room_preferred_deep_depth=6.25,
            semantic_completion_tolerance=0.8,
            visual_two_pose_lateral_enabled=True,
            adaptive_minimal_viewpoints=True)
        scheduler = LightweightRoomScheduler(config)
        door = EstimatedDoorway(
            door_id="f3-open-timeout", center=(0.0, 0.0),
            normal_direction=0.0, width=1.2,
            left_frame_point=(0.0, 0.6), right_frame_point=(0.0, -0.6),
            corridor_side=(-0.6, 0.0), interior_side=(1.1, 0.0),
            confidence=0.9, entered_at=1.0,
            room_id="floor_3_estimated_room_01",
            viewpoint_contract="open_deep_near")
        scheduler.active_door = door
        scheduler.active_room_id = door.room_id
        scheduler.entry_confirmed = True
        scheduler.room_started_at = 1.0
        scheduler.state = "G3_LEFT"
        scheduler.visual_deepening_requested = True
        scheduler.role_attempts = {"G3": 1}
        goal = {
            "source": "lightweight_room_semantic",
            "room_role": "G3", "room_id": door.room_id,
            "position": [6.20, 0.0, 0.0],
            "room_goal_diagnostic": {
                "two_pose_room_sweep": True,
                "two_pose_strategy": "deep_then_axial_near"},
            "execution_result": {"completed_waypoints": 0},
        }

        result = scheduler.record_result(
            goal, False, (5.70, 0.0), 8.0, "room_budget_timeout")

        self.assertTrue(result["success"])
        self.assertEqual(result["reason"],
                         "semantic_observation_neighborhood_reached")
        self.assertIn("G3", scheduler.successful_roles)
        self.assertEqual(door.g3_truth_pose, (5.70, 0.0))

    def test_start_footprint_blocked_refreshes_then_replans_same_role(self):
        """A cold-map self footprint must not consume G3 or advance to G4."""
        grid = OccupancyGrid2D(
            np.zeros((180, 180), dtype=np.int16), 0.10, -9.0, -9.0)
        config = LightweightRoomConfig(
            enabled=True, minimum_goal_separation=1.8,
            semantic_completion_tolerance=0.8,
            visual_two_pose_lateral_enabled=True,
            adaptive_minimal_viewpoints=True,
            adaptive_maximum_viewpoints=1)
        scheduler = LightweightRoomScheduler(config)
        door = EstimatedDoorway(
            door_id="f3-cold-start", center=(0.0, 0.0),
            normal_direction=0.0, width=1.2,
            left_frame_point=(0.0, 0.6), right_frame_point=(0.0, -0.6),
            corridor_side=(-0.6, 0.0), interior_side=(1.1, 0.0),
            confidence=0.9, entered_at=1.0,
            room_id="floor_3_estimated_room_04",
            viewpoint_contract="open_deep_near")
        scheduler.active_door = door
        scheduler.active_room_id = door.room_id
        scheduler.entry_confirmed = True
        scheduler.room_started_at = 1.0
        scheduler.return_anchor = door.interior_side
        scheduler.state = "G3_LEFT"
        scheduler.visual_deepening_requested = True
        scheduler.role_attempts = {"G3": 1}
        rejected_goal = {
            "source": "lightweight_room_semantic",
            "room_role": "G3", "room_id": door.room_id,
            "position": [5.8, 0.0, 0.0],
            "room_goal_diagnostic": {
                "two_pose_room_sweep": True,
                "two_pose_strategy": "deep_then_axial_near"},
            "execution_result": {"completed_waypoints": 0},
        }

        result = scheduler.record_result(
            rejected_goal, False, (1.25, 0.0), 3.0,
            "start_footprint_blocked")

        self.assertFalse(result["success"])
        self.assertEqual(scheduler.visual_geometry_map_refresh_pending, "G3")
        self.assertNotIn("G3", scheduler.role_attempts)
        refresh_goal = scheduler.next_goal(grid, (1.25, 0.0), 4.0)
        self.assertIsNotNone(refresh_goal)
        self.assertEqual(refresh_goal["room_role"], "G3")
        self.assertEqual(refresh_goal["position"][:2], [1.25, 0.0])
        self.assertTrue(refresh_goal["room_goal_diagnostic"][
            "entry_pose_visual_fallback"])
        self.assertEqual(
            refresh_goal["room_goal_diagnostic"]["two_pose_strategy"],
            "start_footprint_map_refresh_before_same_role")

        refresh_result = scheduler.record_result(
            refresh_goal, True, (1.25, 0.0), 5.0,
            "already_at_path_end")
        self.assertTrue(refresh_result["success"])
        self.assertNotIn("G3", scheduler.successful_roles)
        self.assertNotIn("G3", scheduler.role_attempts)

    def test_run12_obstacle_collision_quarantines_exact_g4_endpoint(self):
        scheduler = LightweightRoomScheduler(LightweightRoomConfig(
            enabled=True, minimum_goal_separation=1.8,
            visual_two_pose_lateral_enabled=True,
            adaptive_minimal_viewpoints=True))
        door = EstimatedDoorway(
            door_id="run12-room04", center=(0.0, 0.0),
            normal_direction=0.0, width=1.2,
            left_frame_point=(0.0, 0.6), right_frame_point=(0.0, -0.6),
            corridor_side=(-0.6, 0.0), interior_side=(1.1, 0.0),
            confidence=0.9, entered_at=1.0,
            room_id="floor_3_estimated_room_04",
            viewpoint_contract="obstacle_front_opposite_sides",
            viewpoint_contract_locked_before_entry=True,
            truth_contract_center=(0.0, 0.0),
            truth_contract_normal_direction=0.0,
            truth_obstacle_front_depth_m=3.8)
        scheduler.active_door = door
        scheduler.active_room_id = door.room_id
        scheduler.entry_confirmed = True
        scheduler.room_started_at = 1.0
        scheduler.state = "G4_RIGHT"
        scheduler.completed_points = [(2.425, 0.92)]
        scheduler.successful_roles = {"G3"}
        scheduler.role_attempts = {"G4": 1}
        rejected = [1.85, -1.05, 0.0]
        goal = {
            "source": "lightweight_room_semantic",
            "room_role": "G4", "room_id": door.room_id,
            "position": rejected,
            "room_goal_diagnostic": {
                "two_pose_room_sweep": True,
                "two_pose_strategy":
                    "truth_obstacle_front_gap_opposite_side"},
            "execution_result": {"completed_waypoints": 0},
        }

        result = scheduler.record_result(
            goal, False, (2.425, 0.92), 8.0,
            "scan_lite_refinement_failed")

        self.assertFalse(result["success"])
        self.assertEqual(
            scheduler.failed_visual_geometry_targets["G4"],
            [(1.85, -1.05)])
        self.assertIn(
            "ROOM_OBSTACLE_GAP_ENDPOINT_QUARANTINED",
            [event["event"] for event in scheduler.events])

    def test_open_goal_unreachable_quarantines_role_for_one_reselection(self):
        scheduler = LightweightRoomScheduler(LightweightRoomConfig(
            enabled=True, minimum_goal_separation=1.8,
            visual_two_pose_lateral_enabled=True,
            adaptive_minimal_viewpoints=True))
        door = EstimatedDoorway(
            door_id="open-unreachable", center=(0.0, 0.0),
            normal_direction=0.0, width=1.2,
            left_frame_point=(0.0, 0.6), right_frame_point=(0.0, -0.6),
            corridor_side=(-0.6, 0.0), interior_side=(1.1, 0.0),
            confidence=0.9, entered_at=1.0,
            room_id="estimated_room_03", viewpoint_contract="open_deep_near")
        scheduler.active_door = door
        scheduler.active_room_id = door.room_id
        scheduler.entry_confirmed = True
        scheduler.room_started_at = 1.0
        scheduler.state = "G3_LEFT"
        scheduler.role_attempts = {"G3": 1}
        goal = {
            "source": "lightweight_room_semantic", "room_role": "G3",
            "room_id": door.room_id, "position": [6.7, 0.0, 0.0],
            "room_goal_diagnostic": {
                "two_pose_strategy":
                    "truth_layout_open_centerline_deep_near"},
            "execution_result": {"completed_waypoints": None},
        }

        result = scheduler.record_result(
            goal, False, (1.1, 0.0), 8.0, "goal_unreachable")

        self.assertFalse(result["success"])
        self.assertEqual(
            scheduler.failed_visual_geometry_targets["G3"], [(6.7, 0.0)])
        self.assertNotIn("G3", scheduler.role_attempts)
        self.assertIn("G3", scheduler.visual_geometry_retry_roles)

    def test_two_blocked_deep_endpoints_keep_strict_deep_contract(self):
        """Blocked endpoints are quarantined without weakening a deep room."""
        scheduler = LightweightRoomScheduler(LightweightRoomConfig(
            enabled=True, minimum_goal_separation=1.8,
            visual_two_pose_lateral_enabled=True,
            adaptive_minimal_viewpoints=True))
        door = EstimatedDoorway(
            door_id="run54-f1-lounge", center=(0.0, 0.0),
            normal_direction=0.0, width=1.2,
            left_frame_point=(0.0, 0.6), right_frame_point=(0.0, -0.6),
            corridor_side=(-0.6, 0.0), interior_side=(1.1, 0.0),
            confidence=0.9, entered_at=1.0,
            room_id="estimated_room_01",
            viewpoint_contract="obstacle_front_opposite_sides",
            truth_obstacle_view_policy="deep_outer_side_peek",
            truth_contract_center=(0.0, 0.0),
            truth_contract_normal_direction=0.0,
            truth_obstacle_front_depth_m=3.8,
            truth_obstacle_minimum_lateral_m=-0.3,
            truth_obstacle_maximum_lateral_m=0.3)
        scheduler.active_door = door
        scheduler.active_room_id = door.room_id
        scheduler.entry_confirmed = True
        scheduler.room_started_at = 1.0
        scheduler.state = "G3_LEFT"
        scheduler.role_attempts = {"G3": 1}
        scheduler.failed_visual_geometry_targets = {"G3": [(3.3, 1.5)]}
        goal = {
            "source": "lightweight_room_semantic",
            "room_role": "G3", "room_id": door.room_id,
            "position": [3.3, -1.5, 0.0],
            "room_goal_diagnostic": {
                "two_pose_room_sweep": True,
                "two_pose_strategy":
                    "truth_obstacle_front_gap_opposite_side"},
            "execution_result": {"completed_waypoints": 0},
        }

        scheduler.record_result(
            goal, False, (1.1, 0.0), 8.0, "goal_footprint_blocked")

        self.assertEqual(
            door.truth_obstacle_view_policy, "deep_outer_side_peek")
        self.assertIn(
            "ROOM_DEEP_SIDE_PEEK_REMAINS_STRICT_AFTER_REJECTIONS",
            [event["event"] for event in scheduler.events])
        self.assertEqual(
            sum(len(points) for points in
                scheduler.failed_visual_geometry_targets.values()), 2)
        self.assertEqual(scheduler.adaptive_route_queue, [])

    def test_deep_room_grace_survives_live_shallow_chord_downgrade(self):
        """The replacement G4 keeps the deep room's bounded time reserve."""
        config = LightweightRoomConfig(
            enabled=True, room_budget_seconds=44.0,
            exit_reserve_seconds=10.0,
            deep_obstacle_contract_grace_seconds=40.0,
            minimum_goal_separation=1.8,
            visual_two_pose_lateral_enabled=True,
            adaptive_minimal_viewpoints=True)
        scheduler = LightweightRoomScheduler(config)
        door = EstimatedDoorway(
            door_id="run55-downgraded", center=(0.0, 0.0),
            normal_direction=0.0, width=1.2,
            left_frame_point=(0.0, 0.6), right_frame_point=(0.0, -0.6),
            corridor_side=(-0.6, 0.0), interior_side=(1.1, 0.0),
            confidence=0.9, entered_at=1.0,
            room_id="estimated_room_02",
            viewpoint_contract="obstacle_front_opposite_sides",
            truth_obstacle_view_policy="shallow_front_chord",
            truth_contract_center=(0.0, 0.0),
            truth_contract_normal_direction=0.0,
            truth_obstacle_front_depth_m=3.8,
            truth_obstacle_minimum_lateral_m=-0.3,
            truth_obstacle_maximum_lateral_m=0.3)
        scheduler.active_door = door
        scheduler.active_room_id = door.room_id
        scheduler.entry_confirmed = True
        scheduler.room_started_at = 1.0
        scheduler.deep_obstacle_contract_grace_armed = True
        scheduler.state = "G3_LEFT"
        scheduler.completed_points = [(1.3, -1.0)]
        scheduler.successful_roles = {"G3"}
        scheduler.role_attempts = {"G3": 1}
        grid = OccupancyGrid2D(
            np.zeros((100, 100), dtype=np.int16), 0.10, -5.0, -5.0)

        goal = scheduler.next_goal(grid, (1.3, -1.0), 52.0)

        self.assertNotEqual(goal["room_role"], "EXIT")
        self.assertNotIn("G4", scheduler.failed_roles)

    def test_obstacle_partner_accepts_measured_contract_pose(self):
        """A true opposite gap pose must not fail on raster-target offset."""
        config = LightweightRoomConfig(
            enabled=True, minimum_goal_separation=1.8,
            semantic_completion_tolerance=0.6,
            visual_two_pose_lateral_enabled=True,
            adaptive_minimal_viewpoints=True)
        scheduler = LightweightRoomScheduler(config)
        door = EstimatedDoorway(
            door_id="run55-far-shallow", center=(0.0, 0.0),
            normal_direction=0.0, width=1.2,
            left_frame_point=(0.0, 0.6), right_frame_point=(0.0, -0.6),
            corridor_side=(-0.6, 0.0), interior_side=(1.1, 0.0),
            confidence=0.9, entered_at=1.0,
            room_id="estimated_room_03",
            viewpoint_contract="obstacle_front_opposite_sides",
            truth_obstacle_view_policy="shallow_front_chord",
            truth_contract_center=(0.0, 0.0),
            truth_contract_normal_direction=0.0,
            truth_obstacle_front_depth_m=3.1,
            truth_obstacle_minimum_lateral_m=-0.5,
            truth_obstacle_maximum_lateral_m=0.5)
        scheduler.active_door = door
        scheduler.active_room_id = door.room_id
        scheduler.entry_confirmed = True
        scheduler.room_started_at = 1.0
        scheduler.state = "G4_RIGHT"
        scheduler.completed_points = [(1.3, -1.08)]
        scheduler.successful_roles = {"G3"}
        goal = {
            "source": "lightweight_room_semantic",
            "room_role": "G4", "room_id": door.room_id,
            "position": [1.3, 1.90, 0.0],
            "room_goal_diagnostic": {
                "two_pose_room_sweep": True,
                "two_pose_strategy": "obstacle_front_gap_opposite_side"},
            "execution_result": {
                "completed_waypoints": 2, "reason": "goal_reached"},
        }

        result = scheduler.record_result(
            goal, True, (1.3, 1.08), 8.0, None)

        self.assertTrue(result["success"])
        self.assertEqual(result["reason"], "obstacle_contract_pose_reached")
        self.assertEqual(door.g4_truth_pose, (1.3, 1.08))

    def test_obstacle_partner_rejects_close_target_inside_side_contract(self):
        """Target proximity cannot credit run158's shallow lateral arrival."""
        config = LightweightRoomConfig(
            enabled=True, minimum_goal_separation=1.8,
            semantic_completion_tolerance=0.8,
            visual_two_pose_lateral_enabled=True,
            adaptive_minimal_viewpoints=True)
        scheduler = LightweightRoomScheduler(config)
        door = EstimatedDoorway(
            door_id="run158-deep", center=(0.0, 0.0),
            normal_direction=0.0, width=1.4,
            left_frame_point=(0.0, 0.7), right_frame_point=(0.0, -0.7),
            corridor_side=(-0.6, 0.0), interior_side=(0.75, 0.0),
            confidence=0.9, entered_at=1.0,
            room_id="estimated_room_01",
            viewpoint_contract="obstacle_front_opposite_sides",
            truth_obstacle_view_policy="deep_outer_side_peek",
            truth_contract_center=(0.0, 0.0),
            truth_contract_normal_direction=0.0,
            truth_obstacle_front_depth_m=3.1,
            truth_obstacle_minimum_lateral_m=-0.5,
            truth_obstacle_maximum_lateral_m=0.5)
        scheduler.active_door = door
        scheduler.active_room_id = door.room_id
        scheduler.entry_confirmed = True
        scheduler.room_started_at = 1.0
        scheduler.state = "G4_RIGHT"
        scheduler.completed_points = [(2.55, 1.49)]
        scheduler.successful_roles = {"G3"}
        goal = {
            "source": "lightweight_room_semantic",
            "room_role": "G4", "room_id": door.room_id,
            "position": [2.42, -0.99, 0.0],
            "room_goal_diagnostic": {
                "two_pose_room_sweep": True,
                "two_pose_strategy": "obstacle_front_gap_opposite_side"},
            "execution_result": {
                "completed_waypoints": 3, "reason": "goal_reached"},
        }

        result = scheduler.record_result(
            goal, True, (2.42, -0.90), 8.0, None)

        self.assertFalse(result["success"])
        self.assertEqual(result["reason"],
                         "obstacle_contract_pose_not_reached")
        self.assertIsNone(door.g4_truth_pose)

    def test_deep_obstacle_views_are_bounded_around_primary_blocker(self):
        """A whole-room visibility envelope must not cause run93's 7 m leg."""
        grid = OccupancyGrid2D(
            np.zeros((240, 240), dtype=np.int16), 0.10, -12.0, -12.0)
        config = LightweightRoomConfig(
            enabled=True, goal_clearance=0.38,
            room_depth_probe_range=7.5, room_side_probe_range=7.5,
            maximum_side_lateral=5.0, minimum_goal_separation=1.8,
            semantic_completion_tolerance=0.8)
        door = EstimatedDoorway(
            door_id="f3-room2", center=(0.0, 0.0),
            normal_direction=0.0, width=1.4,
            left_frame_point=(0.0, 0.7),
            right_frame_point=(0.0, -0.7),
            corridor_side=(-0.6, 0.0), interior_side=(1.1, 0.0),
            confidence=0.9, entered_at=1.0,
            viewpoint_contract="obstacle_front_opposite_sides",
            truth_obstacle_view_policy="deep_outer_side_peek",
            truth_contract_center=(0.0, 0.0),
            truth_contract_normal_direction=0.0,
            truth_obstacle_front_depth_m=3.8,
            truth_obstacle_rear_depth_m=4.6,
            truth_obstacle_minimum_lateral_m=-0.3,
            truth_obstacle_maximum_lateral_m=0.3,
            truth_obstacle_visibility_minimum_lateral_m=-3.65,
            truth_obstacle_visibility_maximum_lateral_m=3.45)

        first = select_truth_obstacle_gap_first_visual_view(
            grid, door, (1.1, 0.0), config)
        self.assertIsNotNone(first)
        first_lateral = door.contract_lateral(first["position"])
        self.assertGreaterEqual(abs(first_lateral), 1.45)
        self.assertLessEqual(abs(first_lateral), 1.90)

        template = {
            "source": "lightweight_room_semantic",
            "room_role": "G4", "room_id": "room"}
        second = select_front_gap_opposite_side_second_visual_view(
            grid, door, first["position"], [first["position"]], template,
            config)
        self.assertIsNotNone(second)
        second_lateral = door.contract_lateral(second["position"])
        self.assertLess(first_lateral * second_lateral, 0.0)
        self.assertGreaterEqual(abs(second_lateral), 1.45)
        self.assertLessEqual(abs(second_lateral), 1.90)

    def test_deep_obstacle_pair_crosses_before_nearer_centre_blocker(self):
        """run158's G3->G4 must not graze the near chair diagonally."""
        grid = OccupancyGrid2D(
            np.zeros((240, 240), dtype=np.int16), 0.10, -12.0, -12.0)
        config = LightweightRoomConfig(
            enabled=True, goal_clearance=0.38,
            room_depth_probe_range=7.5, room_side_probe_range=7.5,
            maximum_side_lateral=5.0, minimum_goal_separation=1.8,
            semantic_completion_tolerance=0.8,
            post_entry_path_clearance=0.20)
        door = EstimatedDoorway(
            door_id="f1-room1-near-chair", center=(0.0, 0.0),
            normal_direction=0.0, width=1.4,
            left_frame_point=(0.0, 0.7),
            right_frame_point=(0.0, -0.7),
            corridor_side=(-0.6, 0.0), interior_side=(0.75, 0.0),
            confidence=0.9, entered_at=1.0,
            viewpoint_contract="obstacle_front_opposite_sides",
            truth_obstacle_view_policy="deep_outer_side_peek",
            truth_contract_center=(0.0, 0.0),
            truth_contract_normal_direction=0.0,
            truth_obstacle_front_depth_m=3.1,
            truth_obstacle_nearest_blocker_front_depth_m=1.8,
            truth_obstacle_rear_depth_m=5.3,
            truth_obstacle_minimum_lateral_m=-0.5,
            truth_obstacle_maximum_lateral_m=0.5)
        first = (2.575, 1.56)
        template = {
            "source": "lightweight_room_semantic",
            "room_role": "G4", "room_id": "room"}

        second = select_front_gap_opposite_side_second_visual_view(
            grid, door, first, [first], template, config)

        self.assertIsNotNone(second)
        self.assertTrue(second["near_blocker_segmented_transition"])
        route = second["preflight_path"]
        self.assertEqual(len(route), 3)
        crossing_depth = second["near_blocker_crossing_depth_m"]
        self.assertLess(crossing_depth, 1.8 - 0.30)
        self.assertAlmostEqual(door.contract_depth(route[0]),
                               crossing_depth, places=6)
        self.assertAlmostEqual(door.contract_depth(route[1]),
                               crossing_depth, places=6)
        self.assertGreater(door.contract_lateral(route[0]), 1.0)
        self.assertLess(door.contract_lateral(route[1]), -1.0)
        self.assertEqual(
            second["two_pose_obstacle"]["classification"],
            "occupied_centre_component")

    def test_verified_exit_trace_uses_sparse_route_after_direct_audit_rejects(self):
        manager_path = os.path.join(
            SCRIPTS, "baseline_exploration_manager.py")
        with open(manager_path, encoding="utf-8") as stream:
            source = stream.read()
        branch = source.split("exit_sparse_preflight_fallback = bool(", 1)[1][
            :700]
        self.assertIn("verified_exit_trace_audit", branch)
        self.assertIn("direct_portal_exit_chord_failure_segment", branch)
        self.assertIn("original_exit_anchor_contract", branch)

    def test_truth_gap_first_retry_excludes_blocked_side_and_changes_side(self):
        """A quarantined run91 F3 endpoint must not be submitted again."""
        grid = OccupancyGrid2D(
            np.full((240, 240), -1, dtype=np.int16), 0.10, -12.0, -12.0)
        config = LightweightRoomConfig(
            enabled=True, goal_clearance=0.38,
            room_depth_probe_range=7.5, room_side_probe_range=7.5,
            maximum_side_lateral=5.0, minimum_goal_separation=1.8,
            semantic_completion_tolerance=0.85)
        door = EstimatedDoorway(
            door_id="run91-f3-room2", center=(0.0, 0.0),
            normal_direction=0.0, width=1.4,
            left_frame_point=(0.0, 0.7),
            right_frame_point=(0.0, -0.7),
            corridor_side=(-0.6, 0.0), interior_side=(1.1, 0.0),
            confidence=0.9, entered_at=1.0,
            viewpoint_contract="obstacle_front_opposite_sides",
            truth_obstacle_view_policy="deep_outer_side_peek",
            truth_contract_center=(0.0, 0.0),
            truth_contract_normal_direction=0.0,
            truth_obstacle_front_depth_m=3.8,
            truth_obstacle_rear_depth_m=4.6,
            truth_obstacle_minimum_lateral_m=-0.3,
            truth_obstacle_maximum_lateral_m=0.3,
            truth_obstacle_visibility_minimum_lateral_m=-3.65,
            truth_obstacle_visibility_maximum_lateral_m=3.45)

        first = select_truth_obstacle_gap_first_visual_view(
            grid, door, (1.1, 0.4), config)
        self.assertIsNotNone(first)
        retry = select_truth_obstacle_gap_first_visual_view(
            grid, door, (1.1, 0.4), config,
            excluded_targets=[first["position"]])

        self.assertIsNotNone(retry)
        self.assertGreaterEqual(
            np.linalg.norm(np.asarray(retry["position"]) -
                           np.asarray(first["position"])), 0.35)
        self.assertLess(first["lateral"] * retry["lateral"], 0.0)
        self.assertEqual(retry["excluded_obstacle_gap_target_count"], 1)
        self.assertGreaterEqual(retry["depth"], 3.20)

    def test_room_budget_timeout_quarantines_visual_endpoint(self):
        """A long run93 side peek must not be submitted under G3/G4 again."""
        scheduler = self.scheduler
        door = EstimatedDoorway(
            door_id="run93-deep", center=(0.0, 0.0),
            normal_direction=0.0, width=1.4,
            left_frame_point=(0.0, 0.7), right_frame_point=(0.0, -0.7),
            corridor_side=(-0.6, 0.0), interior_side=(1.1, 0.0),
            confidence=0.9, entered_at=1.0, room_id="room-run93",
            viewpoint_contract="obstacle_front_opposite_sides",
            truth_obstacle_view_policy="deep_outer_side_peek",
            truth_contract_center=(0.0, 0.0),
            truth_contract_normal_direction=0.0,
            truth_obstacle_front_depth_m=3.8,
            truth_obstacle_minimum_lateral_m=-0.3,
            truth_obstacle_maximum_lateral_m=0.3)
        scheduler.active_door = door
        scheduler.active_room_id = door.room_id
        goal = {
            "source": "lightweight_room_semantic",
            "room_role": "G3", "room_id": door.room_id,
            "position": (3.25, 1.65),
            "room_goal_diagnostic": {
                "two_pose_strategy": "obstacle_front_gap_first"},
            "execution_result": {"completed_waypoints": 1}}
        scheduler.record_result(
            goal, False, final_point=(2.0, 0.8), now=10.0,
            reason="room_budget_timeout")
        self.assertIn("G3", scheduler.failed_visual_geometry_targets)
        self.assertEqual(
            scheduler.failed_visual_geometry_targets["G3"][0],
            (3.25, 1.65))

    def test_shallow_generic_obstacle_g4_is_replaced_before_dispatch(self):
        """Run14's 0.67 m G4 must become a deep opposite-side peek."""
        grid = OccupancyGrid2D(
            np.zeros((180, 180), dtype=np.int16), 0.10, -9.0, -9.0)
        config = LightweightRoomConfig(
            enabled=True, goal_clearance=0.38,
            room_depth_probe_range=7.2,
            minimum_goal_separation=1.8,
            semantic_completion_tolerance=0.8,
            visual_two_pose_lateral_enabled=True,
            adaptive_minimal_viewpoints=True)
        scheduler = LightweightRoomScheduler(config)
        door = EstimatedDoorway(
            door_id="run14-shallow-g4", center=(0.0, 0.0),
            normal_direction=0.0, width=1.2,
            left_frame_point=(0.0, 0.6), right_frame_point=(0.0, -0.6),
            corridor_side=(-0.6, 0.0), interior_side=(1.1, 0.0),
            confidence=0.9, entered_at=1.0,
            room_id="estimated_room_02",
            viewpoint_contract="obstacle_front_opposite_sides",
            truth_obstacle_view_policy="deep_outer_side_peek",
            truth_contract_center=(0.0, 0.0),
            truth_contract_normal_direction=0.0,
            truth_obstacle_front_depth_m=3.1,
            truth_obstacle_rear_depth_m=5.3,
            truth_obstacle_minimum_lateral_m=-0.5,
            truth_obstacle_maximum_lateral_m=0.5)
        scheduler.active_door = door
        scheduler.active_room_id = door.room_id
        scheduler.entry_confirmed = True
        scheduler.room_started_at = 1.0
        scheduler.return_anchor = door.interior_side
        scheduler.state = "G4_RIGHT"
        scheduler.visual_deepening_requested = True
        scheduler.completed_points = [(2.55, 1.70)]
        scheduler.successful_roles = {"G3"}
        scheduler.role_attempts = {"G3": 1}
        scheduler.adaptive_route_queue = [{
            "position": (0.67, -2.10), "role": "G4", "score": 0.0,
            "clearance": 1.0, "coverage_cells": set(),
            "adaptive_marginal_gain_m2": 1.0,
            "adaptive_utility": 0.0,
            "preflight_path_length_m": 4.1,
            "preflight_path": [(2.55, 1.70), (0.67, -2.10)],
            "two_pose_room_sweep": True,
            "two_pose_strategy": "obstacle_front_side_split",
            "two_pose_index": 2,
        }]

        goal = scheduler.next_goal(grid, (2.55, 1.70), 5.0)

        self.assertIsNotNone(goal)
        self.assertEqual(goal["room_role"], "G4")
        diagnostic = goal["room_goal_diagnostic"]
        self.assertGreaterEqual(
            door.contract_depth(goal["position"]), 2.10)
        self.assertLess(door.contract_lateral(goal["position"]), -1.20)
        self.assertTrue(diagnostic["obstacle_side_peek_visibility_target"])
        self.assertIn(
            "ROOM_TWO_POSE_OPPOSITE_SIDE_REPAIRED",
            [event["event"] for event in scheduler.events])

    def test_run109_shallow_g4_cannot_repeat_g3_or_leave_front_gap(self):
        """A truth-locked shallow G4 must survive adaptive breadth logic."""
        grid = OccupancyGrid2D(
            np.zeros((180, 180), dtype=np.int16), 0.10, -9.0, -9.0)
        config = LightweightRoomConfig(
            enabled=True, goal_clearance=0.38,
            room_depth_probe_range=7.2,
            minimum_goal_separation=1.8,
            semantic_completion_tolerance=0.8,
            visual_two_pose_lateral_enabled=True,
            adaptive_minimal_viewpoints=True)
        scheduler = LightweightRoomScheduler(config)
        door = EstimatedDoorway(
            door_id="run109-f1-shallow", center=(0.0, 0.0),
            normal_direction=0.0, width=1.2,
            left_frame_point=(0.0, 0.6), right_frame_point=(0.0, -0.6),
            corridor_side=(-0.6, 0.0), interior_side=(1.1, 0.0),
            confidence=0.9, entered_at=1.0,
            room_id="estimated_room_01",
            viewpoint_contract="obstacle_front_opposite_sides",
            truth_obstacle_view_policy="shallow_front_chord",
            truth_contract_center=(0.0, 0.0),
            truth_contract_normal_direction=0.0,
            truth_obstacle_front_depth_m=3.1,
            truth_obstacle_rear_depth_m=5.3,
            truth_obstacle_minimum_lateral_m=-0.5,
            truth_obstacle_maximum_lateral_m=0.5,
            truth_obstacle_nearest_blocker_front_depth_m=1.8)
        scheduler.active_door = door
        scheduler.active_room_id = door.room_id
        scheduler.entry_confirmed = True
        scheduler.room_started_at = 1.0
        scheduler.return_anchor = door.interior_side
        scheduler.state = "G3_LEFT"
        scheduler.visual_deepening_requested = True
        scheduler.completed_points = [(1.11, -1.13)]
        scheduler.successful_roles = {"G3"}
        scheduler.role_attempts = {"G3": 1}

        goal = scheduler.next_goal(grid, (1.11, -1.13), 5.0)

        self.assertIsNotNone(goal)
        self.assertEqual(goal["room_role"], "G4")
        self.assertLess(
            door.contract_lateral(scheduler.completed_points[-1]) *
            door.contract_lateral(goal["position"]), 0.0)
        self.assertLessEqual(door.contract_depth(goal["position"]), 1.32)
        self.assertGreaterEqual(
            np.linalg.norm(np.asarray(goal["position"][:2]) -
                           np.asarray(scheduler.completed_points[-1])), 1.8)
        self.assertNotIn(
            "ADAPTIVE_ROOM_VISUAL_BREADTH_VIEW_SELECTED",
            [event["event"] for event in scheduler.events])

    def test_failed_g3_attempt_does_not_relabel_next_route_as_g4(self):
        """Only physical success, not a failed attempt, advances G3 to G4."""
        grid = OccupancyGrid2D(
            np.zeros((180, 180), dtype=np.int16), 0.10, -9.0, -9.0)
        config = LightweightRoomConfig(
            enabled=True, minimum_goal_separation=1.2,
            visual_two_pose_lateral_enabled=True,
            adaptive_minimal_viewpoints=True,
            adaptive_maximum_viewpoints=1,
            side_goal_retry_limit=2)
        scheduler = LightweightRoomScheduler(config)
        door = EstimatedDoorway(
            door_id="failed-g3-role", center=(0.0, 0.0),
            normal_direction=0.0, width=1.2,
            left_frame_point=(0.0, 0.6), right_frame_point=(0.0, -0.6),
            corridor_side=(-0.6, 0.0), interior_side=(1.1, 0.0),
            confidence=0.9, entered_at=1.0,
            room_id="estimated_room_01",
            viewpoint_contract="open_deep_near")
        scheduler.active_door = door
        scheduler.active_room_id = door.room_id
        scheduler.entry_confirmed = True
        scheduler.room_started_at = 1.0
        scheduler.return_anchor = door.interior_side
        scheduler.state = "G3_LEFT"
        scheduler.visual_deepening_requested = True
        scheduler.visual_entry_refresh_used = 1
        scheduler.role_attempts = {
            "G3": config.side_goal_retry_limit - 1}
        route = {
            "ordered": [{
                "position": (5.8, 0.0), "role": "G4", "score": 0.0,
                "clearance": 1.0, "coverage_cells": set(),
                "adaptive_marginal_gain_m2": 0.0,
                "adaptive_utility": 0.0,
                "preflight_path_length_m": 4.6,
                "preflight_path": [(1.2, 0.0), (5.8, 0.0)],
                "two_pose_room_sweep": True,
                "two_pose_strategy": "truth_layout_open_centerline_deep_near",
            }],
            "strategy": "truth_layout_open_centerline_deep_near",
            "baseline_m": 0.0, "route_length_m": 4.6,
            "return_distance_m": 4.7,
        }

        with mock.patch(
                "lightweight_room_core.plan_two_pose_room_sweep",
                return_value=route):
            goal = scheduler.next_goal(grid, (1.2, 0.0), 5.0)

        self.assertIsNotNone(goal)
        self.assertEqual(goal["room_role"], "G3")
        self.assertNotIn("G4", scheduler.successful_roles)

        scheduler.role_attempts["G3"] = config.side_goal_retry_limit
        scheduler.adaptive_route_queue = []
        with mock.patch(
                "lightweight_room_core.plan_two_pose_room_sweep",
                return_value=route):
            bounded = scheduler.next_goal(grid, (1.2, 0.0), 6.0)
        self.assertTrue(
            bounded is None or bounded.get("room_role") != "G3")

    def test_exhausted_mandatory_g4_releases_room_to_physical_return(self):
        """A failed locked G4 must not be regenerated without a bound."""
        grid = OccupancyGrid2D(
            np.zeros((180, 180), dtype=np.int16), 0.10, -9.0, -9.0)
        config = LightweightRoomConfig(
            enabled=True, minimum_goal_separation=1.8,
            visual_two_pose_lateral_enabled=True,
            adaptive_minimal_viewpoints=True,
            adaptive_maximum_viewpoints=1,
            side_goal_retry_limit=2)
        scheduler = LightweightRoomScheduler(config)
        door = EstimatedDoorway(
            door_id="exhausted-g4", center=(0.0, 0.0),
            normal_direction=0.0, width=1.2,
            left_frame_point=(0.0, 0.6), right_frame_point=(0.0, -0.6),
            corridor_side=(-0.6, 0.0), interior_side=(1.1, 0.0),
            confidence=0.9, entered_at=1.0,
            room_id="estimated_room_01",
            viewpoint_contract="open_deep_near")
        scheduler.active_door = door
        scheduler.active_room_id = door.room_id
        scheduler.entry_confirmed = True
        scheduler.room_started_at = 1.0
        scheduler.return_anchor = door.interior_side
        scheduler.state = "G4_RIGHT"
        scheduler.visual_deepening_requested = True
        scheduler.completed_points = [(6.4, 0.0)]
        scheduler.successful_roles = {"G3"}
        scheduler.role_attempts = {"G3": 1, "G4": 2}
        scheduler.failed_roles = {"G4"}

        goal = scheduler.next_goal(grid, (3.2, 0.0), 20.0)

        self.assertIsNotNone(goal)
        self.assertEqual(goal["room_role"], "RETURN")
        self.assertEqual(scheduler.state, "ROOM_RETURN")
        self.assertIn(
            "ROOM_TWO_POSE_RETRY_EXHAUSTED_RETURN_REQUIRED",
            [event["event"] for event in scheduler.events])

    def test_truth_obstacle_gap_g3_survives_missing_generic_room_polygon(self):
        """A cold F3 room raster must still dispatch its safe gap-side G3."""
        grid = OccupancyGrid2D(
            np.zeros((180, 180), dtype=np.int16), 0.10, -9.0, -9.0)
        config = LightweightRoomConfig(
            enabled=True, goal_clearance=0.38,
            room_depth_probe_range=7.2,
            minimum_goal_separation=1.8,
            visual_two_pose_lateral_enabled=True,
            adaptive_minimal_viewpoints=True,
            adaptive_maximum_viewpoints=1)
        scheduler = LightweightRoomScheduler(config)
        door = EstimatedDoorway(
            door_id="f3-cold-obstacle", center=(0.0, 0.0),
            normal_direction=0.0, width=1.2,
            left_frame_point=(0.0, 0.6), right_frame_point=(0.0, -0.6),
            corridor_side=(-0.6, 0.0), interior_side=(1.1, 0.0),
            confidence=0.9, entered_at=1.0,
            room_id="floor_3_estimated_room_03",
            viewpoint_contract="obstacle_front_opposite_sides",
            truth_obstacle_front_depth_m=3.8,
            truth_obstacle_rear_depth_m=4.6,
            truth_obstacle_minimum_lateral_m=-0.3,
            truth_obstacle_maximum_lateral_m=0.3)
        scheduler.active_door = door
        scheduler.active_room_id = door.room_id
        scheduler.entry_confirmed = True
        scheduler.room_started_at = 1.0
        scheduler.return_anchor = door.interior_side
        scheduler.state = "G1_CENTER"
        scheduler.visual_deepening_requested = True

        with mock.patch(
                "lightweight_room_core.plan_two_pose_room_sweep",
                return_value=None):
            goal = scheduler.next_goal(grid, (1.2, 0.0), 5.0)

        self.assertIsNotNone(goal)
        self.assertEqual(goal["room_role"], "G3")
        diagnostic = goal["room_goal_diagnostic"]
        self.assertEqual(
            diagnostic["two_pose_strategy"], "obstacle_front_gap_first")
        self.assertLess(diagnostic["depth"], 3.8)
        self.assertIn(
            "ROOM_TRUTH_OBSTACLE_GAP_G3_RECOVERED_WITHOUT_ROOM_POLYGON",
            [event["event"] for event in scheduler.events])
        self.assertNotIn(
            "ADAPTIVE_ROOM_OBSERVATION_STOPPED",
            [event["event"] for event in scheduler.events])

    def test_truth_obstacle_gap_pair_uses_bounded_truth_when_grid_unknown(self):
        """A frozen F3 raster still owes two physical front-gap viewpoints."""
        grid = OccupancyGrid2D(
            np.full((180, 180), -1, dtype=np.int16), 0.10, -9.0, -9.0)
        config = LightweightRoomConfig(
            enabled=True, goal_clearance=0.38,
            room_depth_probe_range=7.2,
            minimum_goal_separation=1.2,
            semantic_completion_tolerance=0.4)
        door = EstimatedDoorway(
            door_id="f3-frozen-obstacle", center=(0.0, 0.0),
            normal_direction=0.0, width=1.2,
            left_frame_point=(0.0, 0.6), right_frame_point=(0.0, -0.6),
            corridor_side=(-0.6, 0.0), interior_side=(1.1, 0.0),
            confidence=0.9, entered_at=1.0,
            viewpoint_contract="obstacle_front_opposite_sides",
            truth_obstacle_front_depth_m=3.8,
            truth_obstacle_rear_depth_m=4.6,
            truth_obstacle_minimum_lateral_m=-0.3,
            truth_obstacle_maximum_lateral_m=0.3)

        first = select_truth_obstacle_gap_first_visual_view(
            grid, door, (0.9, 0.0), config)
        self.assertIsNotNone(first)
        self.assertTrue(first["truth_geometry_verified_gap_fallback"])
        self.assertLess(first["depth"], 3.8)
        self.assertGreater(abs(first["lateral"]), 0.8)

        second = select_front_gap_opposite_side_second_visual_view(
            grid, door, first["position"], [first["position"]], {}, config)
        self.assertIsNotNone(second)
        self.assertTrue(second["truth_geometry_verified_gap_fallback"])
        self.assertLess(second["depth"], 3.8)
        self.assertLess(first["lateral"] * second["lateral"], 0.0)
        self.assertGreaterEqual(
            second["visual_baseline_m"],
            config.minimum_goal_separation +
            min(0.10, config.semantic_completion_tolerance))

    def test_truth_obstacle_gap_g3_defers_stale_stripe_to_live_audit(self):
        """Run36's stale F3 stripe must not degrade G3 to a stationary view."""
        occupancy = np.full((180, 180), -1, dtype=np.int16)
        stripe_x = int(round((1.0 - (-9.0)) / 0.10))
        occupancy[55:125, stripe_x] = 100
        grid = OccupancyGrid2D(occupancy, 0.10, -9.0, -9.0)
        config = LightweightRoomConfig(
            enabled=True, goal_clearance=0.38,
            room_depth_probe_range=7.2,
            minimum_goal_separation=1.8,
            semantic_completion_tolerance=0.8)
        door = EstimatedDoorway(
            door_id="run36-f3-stale-g3", center=(0.0, 0.0),
            normal_direction=0.0, width=1.2,
            left_frame_point=(0.0, 0.6), right_frame_point=(0.0, -0.6),
            corridor_side=(-0.6, 0.0), interior_side=(1.1, 0.0),
            confidence=0.9, entered_at=1.0,
            viewpoint_contract="obstacle_front_opposite_sides",
            truth_obstacle_front_depth_m=3.8,
            truth_obstacle_rear_depth_m=4.6,
            truth_obstacle_minimum_lateral_m=-0.3,
            truth_obstacle_maximum_lateral_m=0.3)

        proposal = select_truth_obstacle_gap_first_visual_view(
            grid, door, (0.9, 0.4), config)

        self.assertIsNotNone(proposal)
        self.assertTrue(proposal["stale_raster_segment_veto_overridden"])
        self.assertTrue(proposal["live_3d_audit_required"])
        self.assertEqual(proposal["preflight_path"], [])
        self.assertLess(proposal["lateral"], 0.0)
        self.assertGreaterEqual(proposal["depth"], 3.20)
        self.assertLess(proposal["depth"], 3.8)
        self.assertGreaterEqual(abs(proposal["lateral"]), 1.45)

    def test_shallow_obstacle_pair_stays_before_nearest_blocker_chord(self):
        """F1 room1 must cross the front gap, not route around its chair."""
        grid = OccupancyGrid2D(
            np.zeros((180, 180), dtype=np.int16), 0.10, -9.0, -9.0)
        config = LightweightRoomConfig(
            enabled=True, goal_clearance=0.38,
            room_depth_probe_range=7.2,
            minimum_goal_separation=1.8,
            semantic_completion_tolerance=0.6)
        door = EstimatedDoorway(
            door_id="f1-shallow-obstacle", center=(0.0, 0.0),
            normal_direction=0.0, width=1.2,
            left_frame_point=(0.0, 0.6), right_frame_point=(0.0, -0.6),
            corridor_side=(-0.6, 0.0), interior_side=(1.1, 0.0),
            confidence=0.9, entered_at=1.0,
            viewpoint_contract="obstacle_front_opposite_sides",
            truth_obstacle_view_policy="shallow_front_chord",
            truth_obstacle_front_depth_m=3.1,
            truth_obstacle_rear_depth_m=5.3,
            truth_obstacle_minimum_lateral_m=-0.5,
            truth_obstacle_maximum_lateral_m=0.5,
            truth_obstacle_nearest_blocker_front_depth_m=1.8)

        first = select_truth_obstacle_gap_first_visual_view(
            grid, door, (1.1, 0.0), config)
        self.assertIsNotNone(first)
        second = select_front_gap_opposite_side_second_visual_view(
            grid, door, first["position"], [first["position"]], {}, config)
        self.assertIsNotNone(second)

        self.assertLessEqual(first["depth"], 1.32 + 1e-6)
        self.assertLessEqual(second["depth"], 1.32 + 1e-6)
        self.assertLess(first["lateral"] * second["lateral"], 0.0)
        self.assertLess(abs(first["lateral"]), 1.50)
        self.assertLess(abs(second["lateral"]), 1.50)
        chord = np.linalg.norm(
            np.asarray(first["position"]) - np.asarray(second["position"]))
        self.assertLessEqual(second["preflight_path_length_m"], 1.20 * chord)

        door.g3_truth_pose = first["position"]
        door.g4_truth_pose = second["position"]
        scheduler = LightweightRoomScheduler(config)
        scheduler.active_door = door
        evidence = scheduler._physical_two_view_contract_evidence()
        self.assertTrue(evidence["physical_two_view_contract_met"])
        self.assertEqual(
            evidence["obstacle_view_policy"], "shallow_front_chord")
        self.assertFalse(evidence["deep_side_peek_required"])

    def test_obstacle_pair_uses_truth_contract_frame_not_offset_online_door(self):
        """Run9's 0.43 m door error must not skew the two gap sides."""
        grid = OccupancyGrid2D(
            np.full((180, 180), -1, dtype=np.int16), 0.10, -9.0, -9.0)
        config = LightweightRoomConfig(
            enabled=True, goal_clearance=0.38,
            room_depth_probe_range=7.2,
            minimum_goal_separation=1.8,
            semantic_completion_tolerance=0.8)
        door = EstimatedDoorway(
            door_id="run9-offset-door", center=(0.0, -0.43),
            normal_direction=0.0, width=1.2,
            left_frame_point=(0.0, 0.17),
            right_frame_point=(0.0, -1.03),
            corridor_side=(-0.6, -0.43), interior_side=(1.1, -0.43),
            confidence=0.9, entered_at=1.0,
            viewpoint_contract="obstacle_front_opposite_sides",
            truth_contract_center=(0.0, 0.0),
            truth_contract_normal_direction=0.0,
            truth_obstacle_front_depth_m=3.8,
            truth_obstacle_rear_depth_m=4.6,
            truth_obstacle_minimum_lateral_m=-0.3,
            truth_obstacle_maximum_lateral_m=0.3)

        first = select_truth_obstacle_gap_first_visual_view(
            grid, door, (0.9, -0.43), config)
        self.assertIsNotNone(first)
        first_lateral = door.contract_lateral(first["position"])
        self.assertGreaterEqual(abs(first_lateral), 1.45)
        self.assertLessEqual(abs(first_lateral), 1.75)
        self.assertGreaterEqual(first["depth"], 3.20)
        self.assertTrue(first["obstacle_side_peek_visibility_target"])

        second = select_front_gap_opposite_side_second_visual_view(
            grid, door, first["position"], [first["position"]], {}, config)
        self.assertIsNotNone(second)
        second_lateral = door.contract_lateral(second["position"])
        self.assertLess(first_lateral * second_lateral, 0.0)
        self.assertGreaterEqual(abs(second_lateral), 1.45)
        self.assertLessEqual(abs(second_lateral), 1.95)
        self.assertGreaterEqual(second["depth"], 3.20)
        self.assertGreaterEqual(
            second["visual_baseline_m"],
            config.minimum_goal_separation + 0.10)
        # The measured portal frame is deliberately offset and therefore
        # cannot be the frame used by the truth-locked camera contract.
        self.assertGreater(
            abs(door.lateral(first["position"]) - first_lateral), 0.40)

    def test_obstacle_gap_retry_rotates_away_from_colliding_endpoint(self):
        """Run12 must not submit one live-3D-colliding G4 three times."""
        grid = OccupancyGrid2D(
            np.full((180, 180), -1, dtype=np.int16), 0.10, -9.0, -9.0)
        config = LightweightRoomConfig(
            enabled=True, goal_clearance=0.38,
            room_depth_probe_range=7.2,
            minimum_goal_separation=1.8,
            semantic_completion_tolerance=0.8)
        door = EstimatedDoorway(
            door_id="run12-gap-retry", center=(0.0, -0.43),
            normal_direction=0.0, width=1.2,
            left_frame_point=(0.0, 0.17),
            right_frame_point=(0.0, -1.03),
            corridor_side=(-0.6, -0.43), interior_side=(1.1, -0.43),
            confidence=0.9, entered_at=1.0,
            viewpoint_contract="obstacle_front_opposite_sides",
            truth_contract_center=(0.0, 0.0),
            truth_contract_normal_direction=0.0,
            truth_obstacle_front_depth_m=3.8,
            truth_obstacle_rear_depth_m=4.6,
            truth_obstacle_minimum_lateral_m=-0.3,
            truth_obstacle_maximum_lateral_m=0.3)
        g3 = (2.425, 0.92)

        first = select_front_gap_opposite_side_second_visual_view(
            grid, door, g3, [g3], {}, config)
        self.assertIsNotNone(first)
        retry = select_front_gap_opposite_side_second_visual_view(
            grid, door, g3, [g3], {}, config,
            excluded_targets=[first["position"]])

        self.assertIsNotNone(retry)
        self.assertGreaterEqual(
            np.linalg.norm(np.asarray(retry["position"]) -
                           np.asarray(first["position"])), 0.35)
        self.assertLess(
            door.contract_lateral(g3) *
            door.contract_lateral(retry["position"]), 0.0)
        self.assertLess(
            door.contract_depth(retry["position"]),
            door.truth_obstacle_front_depth_m)
        self.assertEqual(retry["excluded_obstacle_gap_target_count"], 1)

    def test_truth_obstacle_gap_g4_rejects_stale_near_door_stripe(self):
        """A stale F3 stripe must not shrink the locked 3.8 m front gap."""
        occupancy = np.full((180, 180), -1, dtype=np.int16)
        # The run4 failure projected an occupied wall-like stripe around
        # depth 0.79 m across most lateral samples.  Reproduce that geometry
        # in a door frame whose authoritative furniture front is 3.8 m.
        stripe_x = int(round((1.0 - (-9.0)) / 0.10))
        occupancy[55:125, stripe_x] = 100
        grid = OccupancyGrid2D(occupancy, 0.10, -9.0, -9.0)
        config = LightweightRoomConfig(
            enabled=True, goal_clearance=0.38,
            room_depth_probe_range=7.2,
            minimum_goal_separation=1.8,
            semantic_completion_tolerance=0.8)
        door = EstimatedDoorway(
            door_id="f3-stale-stripe", center=(0.0, 0.0),
            normal_direction=0.0, width=1.2,
            left_frame_point=(0.0, 0.6), right_frame_point=(0.0, -0.6),
            corridor_side=(-0.6, 0.0), interior_side=(1.1, 0.0),
            confidence=0.9, entered_at=1.0,
            viewpoint_contract="obstacle_front_opposite_sides",
            truth_obstacle_front_depth_m=3.8,
            truth_obstacle_rear_depth_m=4.6,
            truth_obstacle_minimum_lateral_m=-0.3,
            truth_obstacle_maximum_lateral_m=0.3)

        second = select_front_gap_opposite_side_second_visual_view(
            grid, door, (0.85, 0.0), [(2.17, 0.54)], {}, config)

        self.assertIsNotNone(second)
        self.assertTrue(second["stale_live_front_edge_rejected"])
        self.assertLess(second["depth"], door.truth_obstacle_front_depth_m)
        self.assertLess(door.lateral((2.17, 0.54)) * second["lateral"], 0.0)
        self.assertGreaterEqual(second["visual_baseline_m"], 1.90)
        # Stay in the local door--obstacle chord, but reach the outer side-peek
        # band used to expose hazards hidden behind the near sofa.
        self.assertGreaterEqual(abs(second["lateral"]), 1.45)
        self.assertLessEqual(abs(second["lateral"]), 1.95)
        self.assertGreaterEqual(second["depth"], 3.20)
        self.assertTrue(second["obstacle_side_peek_visibility_target"])

    def test_geometry_reselection_arms_mandatory_execution_grace(self):
        source_path = os.path.join(SCRIPTS, "lightweight_room_core.py")
        with open(source_path, "r", encoding="utf-8") as stream:
            source = stream.read()
        deadline = source[source.index("execution_retry_armed = bool("):
                          source.index("single_physical_view_owned = bool(",
                                       source.index(
                                           "execution_retry_armed = bool("))]
        self.assertIn(
            "any(role in self.visual_geometry_retry_roles", deadline)
        self.assertIn("for role in missing_physical_roles", deadline)

    def test_f3_room2_deep_side_peek_ray_clears_near_sofa(self):
        """The generic geometry target must expose run12's hidden D83 side."""
        grid = OccupancyGrid2D(
            np.full((180, 180), -1, dtype=np.int16), 0.10, -9.0, -9.0)
        config = LightweightRoomConfig(
            enabled=True, goal_clearance=0.38,
            room_depth_probe_range=7.2,
            minimum_goal_separation=1.8,
            semantic_completion_tolerance=0.8)
        door = EstimatedDoorway(
            door_id="f3-room2-side-peek", center=(0.0, 0.0),
            normal_direction=0.0, width=1.2,
            left_frame_point=(0.0, 0.6), right_frame_point=(0.0, -0.6),
            corridor_side=(-0.6, 0.0), interior_side=(1.1, 0.0),
            confidence=0.9, entered_at=1.0,
            viewpoint_contract="obstacle_front_opposite_sides",
            truth_contract_center=(0.0, 0.0),
            truth_contract_normal_direction=0.0,
            truth_obstacle_front_depth_m=3.8,
            truth_obstacle_rear_depth_m=4.6,
            truth_obstacle_minimum_lateral_m=-0.3,
            truth_obstacle_maximum_lateral_m=0.3)

        view = select_truth_obstacle_gap_first_visual_view(
            grid, door, (0.9, 0.0), config)
        self.assertIsNotNone(view)
        viewpoint = view["position"]
        # Run12 F3 room2 expressed in its contract frame: D83 is far down the
        # positive-lateral side.  sofa_b occupies depth [1.6, 3.4] and
        # lateral [2.658, 3.458].  The ray must already be beyond the sofa's
        # rear-depth edge when it enters that lateral interval.
        danger = (3.692, 5.582)
        sofa_near_lateral = 2.658
        ratio = ((sofa_near_lateral - viewpoint[1]) /
                 (danger[1] - viewpoint[1]))
        ray_depth_at_sofa_edge = (
            viewpoint[0] + ratio * (danger[0] - viewpoint[0]))
        self.assertGreater(ray_depth_at_sofa_edge, 3.40)
        self.assertLess(viewpoint[0], door.truth_obstacle_front_depth_m)

    def test_truth_portal_staging_does_not_confirm_entry(self):
        grid = OccupancyGrid2D(
            np.zeros((100, 100), dtype=np.int16), 0.10, -5.0, -5.0)
        scheduler = LightweightRoomScheduler(LightweightRoomConfig(
            enabled=True, minimum_crossing_depth=0.55))
        door = EstimatedDoorway(
            door_id="truth-stage", center=(1.2, 0.0),
            normal_direction=0.0, width=1.4,
            left_frame_point=(1.2, 0.7), right_frame_point=(1.2, -0.7),
            corridor_side=(0.6, 0.0), interior_side=(2.7, 0.0),
            confidence=0.9, entered_at=1.0)

        active = scheduler.activate_truth_portal_staging(
            grid, door, (-0.5, 0.0), 2.0)

        self.assertIsNotNone(active)
        self.assertFalse(scheduler.entry_confirmed)
        self.assertEqual(scheduler.room_count, 0)
        self.assertTrue(scheduler.prepared_entry[
            "truth_geometry_verified_staging"])
        self.assertEqual(len(scheduler.entry_portal_waypoints), 3)

    def test_truth_portal_staging_allows_exhausted_missing_map_anchor(self):
        grid = OccupancyGrid2D(
            np.full((100, 100), -1, dtype=np.int16), 0.10, -5.0, -5.0)
        scheduler = LightweightRoomScheduler(LightweightRoomConfig(
            enabled=True, minimum_crossing_depth=0.55))
        door = EstimatedDoorway(
            door_id="truth-stage-frozen-map", center=(1.2, 0.0),
            normal_direction=0.0, width=1.4,
            left_frame_point=(1.2, 0.7), right_frame_point=(1.2, -0.7),
            corridor_side=(0.6, 0.0), interior_side=(2.7, 0.0),
            confidence=0.9, entered_at=1.0)

        active = scheduler.activate_truth_portal_staging(
            grid, door, (0.0, 0.0), 2.0)

        self.assertIsNotNone(active)
        self.assertFalse(scheduler.entry_confirmed)
        self.assertEqual(scheduler.room_count, 0)
        self.assertEqual(
            scheduler.events[-1]["candidate_entry_reason"],
            "candidate_no_observed_free_anchor")

    def test_truth_portal_staging_skips_redundant_corridor_anchor_when_beside_door(self):
        grid = OccupancyGrid2D(
            np.zeros((100, 100), dtype=np.int16), 0.10, -5.0, -5.0)
        scheduler = LightweightRoomScheduler(LightweightRoomConfig(
            enabled=True, minimum_crossing_depth=0.55))
        door = EstimatedDoorway(
            door_id="truth-stage-near", center=(1.2, 0.0),
            normal_direction=0.0, width=1.4,
            left_frame_point=(1.2, 0.7), right_frame_point=(1.2, -0.7),
            corridor_side=(0.6, 0.0), interior_side=(2.7, 0.0),
            confidence=0.9, entered_at=1.0)

        active = scheduler.activate_truth_portal_staging(
            grid, door, (0.2, 0.15), 2.0)

        self.assertIsNotNone(active)
        self.assertEqual(len(scheduler.entry_portal_waypoints), 2)
        self.assertEqual(scheduler.entry_portal_waypoints[0], [1.2, 0.0])
        self.assertTrue(scheduler.events[-1]["corridor_anchor_skipped"])

    def test_truth_portal_staging_aligns_far_row_before_lateral_approach(self):
        grid = OccupancyGrid2D(
            np.zeros((400, 100), dtype=np.int16), 0.10, -5.0, 0.0)
        scheduler = LightweightRoomScheduler(LightweightRoomConfig(
            enabled=True, minimum_crossing_depth=0.55))
        door = EstimatedDoorway(
            door_id="truth-stage-far-row", center=(1.19, 28.895),
            normal_direction=0.0, width=1.4,
            left_frame_point=(1.19, 29.595),
            right_frame_point=(1.19, 28.195),
            corridor_side=(0.59, 28.895),
            interior_side=(2.69, 28.895), confidence=0.9,
            entered_at=1.0)

        active = scheduler.activate_truth_portal_staging(
            grid, door, (-0.32, 27.36), 2.0)

        self.assertIsNotNone(active)
        waypoints = scheduler.entry_portal_waypoints
        self.assertEqual(len(waypoints), 4)
        self.assertAlmostEqual(waypoints[0][0], -0.32, places=6)
        self.assertAlmostEqual(waypoints[0][1], 28.895, places=6)
        self.assertTrue(scheduler.events[-1][
            "station_alignment_inserted"])

    def test_truth_obstacle_contract_builds_door_gap_opposite_pair(self):
        grid = OccupancyGrid2D(
            np.zeros((180, 180), dtype=np.int16), 0.10, -9.0, -9.0)
        config = LightweightRoomConfig(
            enabled=True, room_depth_probe_range=7.2,
            maximum_side_lateral=5.0, minimum_goal_separation=1.8,
            visual_breadth_min_baseline=3.6,
            obstacle_front_opposite_side_pair=True)
        door = EstimatedDoorway(
            door_id="estimated_f1_room0", center=(0.0, 0.0),
            normal_direction=0.0, width=1.2,
            left_frame_point=(0.0, 0.6), right_frame_point=(0.0, -0.6),
            corridor_side=(-0.6, 0.0), interior_side=(1.1, 0.0),
            confidence=0.9, entered_at=1.0,
            viewpoint_contract="obstacle_front_opposite_sides",
            viewpoint_contract_source="generated_floor_layout_physical_geometry",
            truth_room_id="floor_0_room_0",
            truth_obstacle_front_depth_m=3.8,
            truth_obstacle_rear_depth_m=4.6,
            truth_obstacle_minimum_lateral_m=-0.3,
            truth_obstacle_maximum_lateral_m=0.3)

        route = plan_two_pose_room_sweep(
            grid, door, (1.5, 0.0), (1.1, 0.0), config)

        self.assertIsNotNone(route)
        self.assertEqual(route["strategy"], "obstacle_front_side_split")
        first, second = route["ordered"]
        self.assertLess(first["lateral"] * second["lateral"], 0.0)
        self.assertLess(max(first["depth"], second["depth"]), 3.8)

    def test_truth_obstacle_gap_g3_survives_narrow_live_door_estimate(self):
        grid = OccupancyGrid2D(
            np.zeros((180, 180), dtype=np.int16), 0.10, -9.0, -9.0)
        config = LightweightRoomConfig(
            enabled=True, goal_clearance=0.38,
            maximum_side_lateral=5.0)
        door = EstimatedDoorway(
            door_id="narrow-truth-gap", center=(0.0, 0.0),
            normal_direction=0.0, width=0.60,
            left_frame_point=(0.0, 0.30),
            right_frame_point=(0.0, -0.30),
            corridor_side=(-0.6, 0.0), interior_side=(0.85, 0.0),
            confidence=0.9, entered_at=1.0,
            viewpoint_contract="obstacle_front_opposite_sides",
            truth_obstacle_front_depth_m=3.1,
            truth_obstacle_rear_depth_m=5.3,
            truth_obstacle_minimum_lateral_m=-0.5,
            truth_obstacle_maximum_lateral_m=0.5)

        proposal = select_truth_obstacle_gap_first_visual_view(
            grid, door, (0.85, 0.0), config)

        self.assertIsNotNone(proposal)
        self.assertLess(proposal["depth"], 3.1)
        self.assertGreater(abs(proposal["lateral"]), 0.5)
        self.assertEqual(proposal["path_planning_clearance_m"], 0.20)

    def test_truth_obstacle_gap_g4_uses_physical_not_open_room_baseline(self):
        grid = OccupancyGrid2D(
            np.zeros((180, 180), dtype=np.int16), 0.10, -9.0, -9.0)
        config = LightweightRoomConfig(
            enabled=True, goal_clearance=0.38,
            room_depth_probe_range=7.2,
            maximum_side_lateral=5.0,
            minimum_goal_separation=1.8,
            visual_breadth_min_baseline=3.6)
        door = EstimatedDoorway(
            door_id="f1-room1-gap", center=(0.0, 0.0),
            normal_direction=0.0, width=0.60,
            left_frame_point=(0.0, 0.30),
            right_frame_point=(0.0, -0.30),
            corridor_side=(-0.6, 0.0), interior_side=(0.85, 0.0),
            confidence=0.9, entered_at=1.0,
            viewpoint_contract="obstacle_front_opposite_sides",
            truth_obstacle_front_depth_m=3.1,
            truth_obstacle_rear_depth_m=5.3,
            truth_obstacle_minimum_lateral_m=-0.5,
            truth_obstacle_maximum_lateral_m=0.5)
        g3 = (2.38, 1.10)

        proposal = select_front_gap_opposite_side_second_visual_view(
            grid, door, g3, [g3], {}, config)

        self.assertIsNotNone(proposal)
        self.assertLess(proposal["lateral"], -0.55)
        self.assertLess(proposal["depth"], 3.1)
        # The strict 1.8 m contract is checked from executed Gazebo poses.  The
        # new endpoint also reaches the outer side-peek band, so its baseline
        # is intentionally wider than the old near-door pair while remaining
        # a short chord wholly before the obstacle front.
        self.assertGreaterEqual(proposal["visual_baseline_m"], 1.90)
        self.assertLess(proposal["visual_baseline_m"], 3.40)
        self.assertTrue(proposal["obstacle_side_peek_visibility_target"])

        # If the nominal G3 timed out but G4 physically reached its side,
        # repair the opposite side as the still-missing G3.  Reissuing G4
        # here caused F3 to leave the room as ENTERED_PARTIAL and later
        # register the already explored portal as a fifth room.
        reverse = select_front_gap_opposite_side_second_visual_view(
            grid, door, proposal["position"], [proposal["position"]],
            {}, config, missing_role="G3")
        self.assertIsNotNone(reverse)
        self.assertEqual(reverse["role"], "G3")
        self.assertEqual(reverse["two_pose_index"], 1)
        self.assertGreater(reverse["lateral"], 0.55)
        self.assertLess(reverse["depth"], 3.1)

    def test_truth_obstacle_generic_g4_behind_front_is_rejected(self):
        door = EstimatedDoorway(
            door_id="f3-room4-gap", center=(0.0, 0.0),
            normal_direction=0.0, width=0.60,
            left_frame_point=(0.0, 0.30),
            right_frame_point=(0.0, -0.30),
            corridor_side=(-0.6, 0.0), interior_side=(0.85, 0.0),
            confidence=0.9, entered_at=1.0,
            viewpoint_contract="obstacle_front_opposite_sides",
            truth_obstacle_front_depth_m=3.8)

        self.assertFalse(obstacle_candidate_is_behind_front_gap(
            door, (2.30, -1.10), 0.20))
        # This reproduces f3nearfirst room04's provisional G4: the lateral
        # sign is opposite, but depth 5.44 is already behind the table.
        self.assertTrue(obstacle_candidate_is_behind_front_gap(
            door, (5.44, -1.61), 0.20))

    def test_deep_room_pose_keeps_bounded_return_before_exit(self):
        door = EstimatedDoorway(
            door_id="f3-room4-exit", center=(0.0, 0.0),
            normal_direction=0.0, width=0.60,
            left_frame_point=(0.0, 0.30),
            right_frame_point=(0.0, -0.30),
            corridor_side=(-0.6, 0.0), interior_side=(0.85, 0.0),
            confidence=0.9, entered_at=1.0)

        self.assertTrue(direct_exit_fold_allowed(door, (2.2, 1.0), 1.2))
        self.assertFalse(direct_exit_fold_allowed(door, (5.0, 1.0), 1.2))

    def test_obstacle_front_gap_g4_folds_directly_into_verified_exit(self):
        door = EstimatedDoorway(
            door_id="f3-room2-gap-exit", center=(0.0, 0.0),
            normal_direction=0.0, width=1.2,
            left_frame_point=(0.0, 0.6), right_frame_point=(0.0, -0.6),
            corridor_side=(-0.6, 0.0), interior_side=(1.1, 0.0),
            confidence=0.9, entered_at=1.0,
            viewpoint_contract="obstacle_front_opposite_sides",
            truth_obstacle_front_depth_m=3.8,
            truth_obstacle_minimum_lateral_m=-0.3,
            truth_obstacle_maximum_lateral_m=0.3)

        self.assertTrue(direct_exit_fold_allowed(
            door, (3.30, -1.50), 1.2))
        self.assertFalse(direct_exit_fold_allowed(
            door, (4.10, -1.50), 1.2))

    def test_executed_open_contract_requires_real_deep_near_geometry(self):
        config = LightweightRoomConfig(
            enabled=True, minimum_goal_separation=1.8,
            open_room_minimum_deep_depth=6.25,
            open_room_preferred_deep_depth=6.25,
            semantic_completion_tolerance=0.8)
        scheduler = LightweightRoomScheduler(config)
        door = EstimatedDoorway(
            door_id="open-contract", center=(0.0, 0.0),
            normal_direction=0.0, width=1.2,
            left_frame_point=(0.0, 0.6), right_frame_point=(0.0, -0.6),
            corridor_side=(-0.6, 0.0), interior_side=(1.1, 0.0),
            confidence=0.9, entered_at=1.0,
            viewpoint_contract="open_deep_near",
            g3_truth_pose=(6.2, 0.0), g4_truth_pose=(2.6, 0.3))
        scheduler.active_door = door

        evidence = scheduler._physical_two_view_contract_evidence()
        self.assertTrue(evidence["physical_two_view_contract_met"])
        self.assertTrue(evidence["depth_delta_met"])
        self.assertTrue(evidence["deep_view_met"])
        self.assertTrue(evidence["near_view_met"])

        door.g4_truth_pose = (4.04, 0.8)
        evidence = scheduler._physical_two_view_contract_evidence()
        self.assertTrue(evidence["physical_two_view_contract_met"])
        self.assertTrue(evidence["near_view_met"])

        door.g4_truth_pose = (4.6, 1.0)
        evidence = scheduler._physical_two_view_contract_evidence()
        self.assertFalse(evidence["physical_two_view_contract_met"])
        self.assertFalse(evidence["near_view_met"])

    def test_executed_open_contract_accepts_bounded_deep_arrival_tolerance(self):
        """The measured run pose may stop just short of its deep target."""
        scheduler = LightweightRoomScheduler(LightweightRoomConfig(
            enabled=True, minimum_goal_separation=1.8,
            open_room_minimum_deep_depth=6.25,
            open_room_preferred_deep_depth=6.25,
            semantic_completion_tolerance=0.8))
        door = EstimatedDoorway(
            door_id="open-tolerated-arrival", center=(0.0, 0.0),
            normal_direction=0.0, width=1.2,
            left_frame_point=(0.0, 0.6), right_frame_point=(0.0, -0.6),
            corridor_side=(-0.6, 0.0), interior_side=(1.1, 0.0),
            confidence=0.9, entered_at=1.0,
            viewpoint_contract="open_deep_near",
            g3_truth_pose=(5.42, 0.0), g4_truth_pose=(2.60, 0.0))
        scheduler.active_door = door

        evidence = scheduler._physical_two_view_contract_evidence()

        self.assertTrue(evidence["physical_two_view_contract_met"])
        self.assertAlmostEqual(evidence["minimum_deep_depth_m"], 5.30)

    def test_partial_portal_cannot_be_registered_again_during_retry_cooldown(self):
        grid = OccupancyGrid2D(
            np.zeros((80, 80), dtype=np.int16), 0.10, -4.0, -4.0)
        scheduler = LightweightRoomScheduler(LightweightRoomConfig(
            enabled=True, duplicate_door_radius=1.5))
        existing = EstimatedDoorway(
            door_id="estimated_door_01", center=(0.0, 0.0),
            normal_direction=0.0, width=1.2,
            left_frame_point=(0.0, 0.6), right_frame_point=(0.0, -0.6),
            corridor_side=(-0.6, 0.0), interior_side=(1.1, 0.0),
            confidence=0.9, entered_at=1.0,
            room_id="estimated_room_01", partial_retry_count=1,
            temporarily_failed=True)
        scheduler.detector.doors.append(existing)
        scheduler._set_door_cooldown(existing.center, 10.0)
        duplicate = EstimatedDoorway(
            door_id="candidate", center=(0.02, -0.01),
            normal_direction=0.0, width=1.2,
            left_frame_point=(0.02, 0.59),
            right_frame_point=(0.02, -0.61),
            corridor_side=(-0.58, -0.01), interior_side=(1.12, -0.01),
            confidence=0.9, entered_at=5.0)

        activated = scheduler._prepare_and_activate(
            grid, duplicate, (-0.6, 0.0), 5.0, proactive=True)

        self.assertIsNone(activated)
        self.assertEqual(len(scheduler.detector.doors), 1)
        self.assertEqual(
            scheduler.events[-1]["event"],
            "DUPLICATE_PARTIAL_DOOR_COOLDOWN_HELD")

    def test_executed_obstacle_contract_requires_front_gap_opposite_sides(self):
        scheduler = LightweightRoomScheduler(LightweightRoomConfig(
            enabled=True, minimum_goal_separation=1.8))
        door = EstimatedDoorway(
            door_id="obstacle-contract", center=(0.0, 0.0),
            normal_direction=0.0, width=1.2,
            left_frame_point=(0.0, 0.6), right_frame_point=(0.0, -0.6),
            corridor_side=(-0.6, 0.0), interior_side=(1.1, 0.0),
            confidence=0.9, entered_at=1.0,
            viewpoint_contract="obstacle_front_opposite_sides",
            truth_obstacle_front_depth_m=3.8,
            truth_obstacle_rear_depth_m=4.6,
            truth_obstacle_minimum_lateral_m=-0.3,
            truth_obstacle_maximum_lateral_m=0.3,
            g3_truth_pose=(3.15, 1.4), g4_truth_pose=(3.15, -1.4))
        scheduler.active_door = door

        evidence = scheduler._physical_two_view_contract_evidence()
        self.assertTrue(evidence["physical_two_view_contract_met"])
        self.assertTrue(evidence["both_views_in_front_gap"])
        self.assertTrue(evidence["opposite_obstacle_sides"])
        self.assertTrue(evidence["both_views_deep_side_peek"])
        self.assertTrue(evidence["outer_side_peek"])

        # Run12's shallow opposite-side pair is no longer accepted merely
        # because it has enough Euclidean baseline.
        door.g3_truth_pose = (2.2, 1.0)
        door.g4_truth_pose = (2.2, -1.0)
        evidence = scheduler._physical_two_view_contract_evidence()
        self.assertFalse(evidence["physical_two_view_contract_met"])
        self.assertFalse(evidence["both_views_deep_side_peek"])

        door.g4_truth_pose = (4.2, -1.0)
        evidence = scheduler._physical_two_view_contract_evidence()
        self.assertFalse(evidence["physical_two_view_contract_met"])
        self.assertFalse(evidence["both_views_in_front_gap"])

        door.g3_truth_pose = (3.15, 1.4)
        door.g4_truth_pose = (3.15, 0.8)
        evidence = scheduler._physical_two_view_contract_evidence()
        self.assertFalse(evidence["physical_two_view_contract_met"])
        self.assertFalse(evidence["opposite_obstacle_sides"])

    def test_deep_obstacle_contract_uses_primary_blocker_not_room_envelope(self):
        scheduler = LightweightRoomScheduler(LightweightRoomConfig(
            enabled=True, minimum_goal_separation=1.8,
            semantic_completion_tolerance=0.8))
        door = EstimatedDoorway(
            door_id="run94-deep-contract", center=(0.0, 0.0),
            normal_direction=0.0, width=1.4,
            left_frame_point=(0.0, 0.7), right_frame_point=(0.0, -0.7),
            corridor_side=(-0.6, 0.0), interior_side=(1.5, 0.0),
            confidence=0.9, entered_at=1.0,
            viewpoint_contract="obstacle_front_opposite_sides",
            truth_obstacle_view_policy="deep_outer_side_peek",
            truth_obstacle_front_depth_m=3.8,
            truth_obstacle_rear_depth_m=4.6,
            truth_obstacle_minimum_lateral_m=-0.3,
            truth_obstacle_maximum_lateral_m=0.3,
            truth_obstacle_visibility_minimum_lateral_m=-3.657,
            truth_obstacle_visibility_maximum_lateral_m=3.457,
            g3_truth_pose=(3.183, -1.396),
            g4_truth_pose=(3.308, 1.355))
        scheduler.active_door = door

        evidence = scheduler._physical_two_view_contract_evidence()
        self.assertTrue(evidence["physical_two_view_contract_met"])
        self.assertTrue(evidence["outer_side_peek"])
        self.assertEqual(
            evidence["room_visibility_envelope_minimum_lateral_m"],
            -3.657)
        self.assertEqual(
            evidence["room_visibility_envelope_maximum_lateral_m"],
            3.457)

    def test_run61_obstacle_contract_accepts_bounded_arrival_tolerance(self):
        scheduler = LightweightRoomScheduler(LightweightRoomConfig(
            enabled=True, minimum_goal_separation=1.8,
            semantic_completion_tolerance=0.8))
        door = EstimatedDoorway(
            door_id="run61-obstacle-contract", center=(0.0, 0.0),
            normal_direction=0.0, width=1.2,
            left_frame_point=(0.0, 0.6), right_frame_point=(0.0, -0.6),
            corridor_side=(-0.6, 0.0), interior_side=(1.1, 0.0),
            confidence=0.9, entered_at=1.0,
            viewpoint_contract="obstacle_front_opposite_sides",
            truth_obstacle_front_depth_m=3.8,
            truth_obstacle_rear_depth_m=4.6,
            truth_obstacle_minimum_lateral_m=-0.3,
            truth_obstacle_maximum_lateral_m=0.3,
            g3_truth_pose=(3.07, 1.44),
            g4_truth_pose=(2.767, -1.12))
        scheduler.active_door = door

        evidence = scheduler._physical_two_view_contract_evidence()

        self.assertTrue(evidence["physical_two_view_contract_met"])
        self.assertTrue(evidence["both_views_deep_side_peek"])
        self.assertAlmostEqual(
            evidence["visibility_arrival_tolerance_m"], 0.20)
        self.assertAlmostEqual(
            evidence["minimum_visibility_depth_m"], 2.60)

        door.g3_truth_pose = (2.2, 1.0)
        door.g4_truth_pose = (2.2, -1.0)
        evidence = scheduler._physical_two_view_contract_evidence()
        self.assertFalse(evidence["physical_two_view_contract_met"])
        self.assertFalse(evidence["both_views_deep_side_peek"])

    @mock.patch("lightweight_room_core.prepare_portal_path")
    @mock.patch("lightweight_room_core.select_room_goal")
    @mock.patch("lightweight_room_core.doorway_candidate_entry_ready")
    def test_validated_fourth_door_survives_exit_centerline_window(
            self, entry_ready, select_goal, prepare_portal):
        """Run13's validated fourth door must reactivate after recentering."""
        scheduler = LightweightRoomScheduler(LightweightRoomConfig(
            enabled=True))
        grid = OccupancyGrid2D(
            np.zeros((100, 100), dtype=np.int16), 0.10, -5.0, -5.0)
        previous = EstimatedDoorway(
            door_id="estimated_door_03", center=(0.0, 0.0),
            normal_direction=0.0, width=1.2,
            left_frame_point=(0.0, 0.6), right_frame_point=(0.0, -0.6),
            corridor_side=(-0.6, 0.0), interior_side=(1.1, 0.0),
            confidence=0.9, entered_at=1.0, room_id="estimated_room_03",
            room_transaction_complete=True)
        fourth = EstimatedDoorway(
            door_id="estimated_door_04", center=(2.0, 0.0),
            normal_direction=math.pi / 2.0, width=1.2,
            left_frame_point=(1.4, 0.0), right_frame_point=(2.6, 0.0),
            corridor_side=(2.0, -0.6), interior_side=(2.0, 1.1),
            confidence=0.9, entered_at=2.0)
        proposal = {
            "position": (2.0, 1.1), "role": "ENTRY", "score": 0.0,
            "depth": 1.1, "lateral": 0.0, "clearance": 1.0,
        }
        portal = {
            "mandatory_portal_waypoints": [(2.0, 0.0), (2.0, 1.1)]}
        scheduler.pending_corridor_recovery_door = previous
        entry_ready.return_value = (True, "ready")
        select_goal.return_value = proposal
        prepare_portal.return_value = portal
        scheduler._activate = mock.Mock(side_effect=[None, fourth])

        activated = scheduler._prepare_and_activate(
            grid, fourth, (1.5, 0.0), 248.073, proactive=True)

        self.assertIsNone(activated)
        self.assertIs(scheduler.deferred_prevalidated_door_after_centerline,
                      fourth)
        self.assertEqual(
            scheduler.events[-2]["event"],
            "VALIDATED_DOOR_DEFERRED_UNTIL_CENTERLINE")

        recovered = scheduler.confirm_corridor_centerline_recovered(
            249.186, (1.8, 0.0), "unit_test")
        self.assertTrue(recovered)
        self.assertIs(scheduler.active_door, None)
        # _activate is mocked, so confirm stores the prepared transaction and
        # emits the reactivation proof without mutating active_door itself.
        self.assertEqual(scheduler.prepared_entry["position"], (2.0, 1.1))
        self.assertEqual(
            scheduler.events[-1]["event"],
            "LOCAL_DOOR_REACTIVATED_AFTER_CENTERLINE")

    def test_medium_two_ray_fragment_does_not_replace_deep_near_pair(self):
        data = np.zeros((160, 160), dtype=np.int16)
        resolution, origin = 0.10, -8.0
        # About 0.45 m2 and only two centre rays: representative of the
        # transient component that misclassified fix1 room 0.
        for wx in np.arange(3.00, 3.91, resolution):
            for wy in np.arange(-0.40, 0.11, resolution):
                data[int((wy - origin) / resolution),
                     int((wx - origin) / resolution)] = 100
        grid = OccupancyGrid2D(data, resolution, origin, origin)
        config = LightweightRoomConfig(
            enabled=True, room_depth_probe_range=6.5,
            maximum_side_lateral=5.0, minimum_goal_separation=1.8,
            visual_breadth_min_baseline=3.6,
            open_room_preferred_deep_depth=4.75,
            open_room_minimum_deep_depth=4.50,
            obstacle_second_view_near_door=True,
            obstacle_front_opposite_side_pair=True,
            obstacle_front_minimum_area_m2=0.55)
        door = EstimatedDoorway(
            door_id="estimated_door_medium_fragment", center=(0.0, 0.0),
            normal_direction=0.0, width=1.2,
            left_frame_point=(0.0, 0.6), right_frame_point=(0.0, -0.6),
            corridor_side=(-0.6, 0.0), interior_side=(1.1, 0.0),
            confidence=0.9, entered_at=1.0)

        route = plan_two_pose_room_sweep(
            grid, door, (1.5, 0.0), (1.1, 0.0), config)

        self.assertIsNotNone(route)
        self.assertEqual(route["strategy"], "deep_then_axial_near")
        first, second = route["ordered"]
        self.assertGreaterEqual(first["depth"], 4.45)
        self.assertGreater(first["depth"], second["depth"])

    def test_sparse_wide_front_face_still_uses_door_gap_pair(self):
        data = np.zeros((160, 160), dtype=np.int16)
        resolution, origin = 0.10, -8.0
        # Only about 0.27 m2 survives projection, but the component spans a
        # coherent one-metre furniture face across the doorway normal.
        for wx in np.arange(2.50, 2.81, resolution):
            for wy in np.arange(-0.50, 0.51, resolution):
                data[int((wy - origin) / resolution),
                     int((wx - origin) / resolution)] = 100
        grid = OccupancyGrid2D(data, resolution, origin, origin)
        config = LightweightRoomConfig(
            enabled=True, room_depth_probe_range=6.5,
            maximum_side_lateral=5.0, minimum_goal_separation=1.8,
            visual_breadth_min_baseline=3.6,
            open_room_preferred_deep_depth=4.75,
            open_room_minimum_deep_depth=4.50,
            obstacle_second_view_near_door=True,
            obstacle_front_opposite_side_pair=True,
            obstacle_front_minimum_area_m2=0.55)
        door = EstimatedDoorway(
            door_id="estimated_door_sparse_wide", center=(0.0, 0.0),
            normal_direction=0.0, width=1.2,
            left_frame_point=(0.0, 0.6), right_frame_point=(0.0, -0.6),
            corridor_side=(-0.6, 0.0), interior_side=(1.1, 0.0),
            confidence=0.9, entered_at=1.0)

        route = plan_two_pose_room_sweep(
            grid, door, (1.5, 0.0), (1.1, 0.0), config)

        self.assertIsNotNone(route)
        self.assertEqual(route["strategy"], "obstacle_front_side_split")
        self.assertTrue(route["obstacle"][
            "broad_coherent_sparse_projection"])
        first, second = route["ordered"]
        self.assertLess(first["lateral"] * second["lateral"], 0.0)
        self.assertLess(max(first["depth"], second["depth"]), 2.50)

    def test_relaxed_uncertain_open_route_stages_near_view_from_live_pose(self):
        grid = OccupancyGrid2D(
            np.zeros((160, 160), dtype=np.int16), 0.10, -8.0, -8.0)
        config = LightweightRoomConfig(
            enabled=True, goal_clearance=0.38,
            post_entry_path_clearance=0.20)
        current = (1.2, 0.0)
        plan = {
            "strategy": "deep_then_blind_near",
            "ordered": [
                {"position": (5.0, -1.5), "depth": 5.0,
                 "lateral": -1.5, "role": "G3",
                 "two_pose_room_sweep": True,
                 "post_entry_path_clearance_relaxed": True},
                {"position": (2.1, 1.5), "depth": 2.1,
                 "lateral": 1.5, "role": "G4",
                 "two_pose_room_sweep": True,
                 "preflight_path": [(5.0, -1.5), (2.1, 1.5)]},
            ],
            "baseline_m": 4.2, "route_length_m": 12.0,
            "return_distance_m": 3.0, "obstacle": None,
        }

        staged, changed = stage_uncertain_open_near_mapping_view(
            grid, current, plan, config, [])

        self.assertTrue(changed)
        self.assertEqual(staged["strategy"],
                         "uncertain_open_near_mapping_view")
        self.assertEqual(len(staged["ordered"]), 1)
        view = staged["ordered"][0]
        self.assertEqual(view["role"], "G3")
        self.assertEqual(view["position"], (2.1, 1.5))
        self.assertTrue(view["replan_second_view_from_fresh_grid"])
        self.assertAlmostEqual(view["preflight_path"][0][0], current[0])
        self.assertAlmostEqual(view["preflight_path"][0][1], current[1])

    def test_small_central_table_keeps_one_deep_view(self):
        data = np.zeros((160, 160), dtype=np.int16)
        resolution, origin = 0.10, -8.0
        for wx in np.arange(3.80, 4.41, resolution):
            for wy in np.arange(-0.30, 0.31, resolution):
                data[int((wy - origin) / resolution),
                     int((wx - origin) / resolution)] = 100
        grid = OccupancyGrid2D(data, resolution, origin, origin)
        config = LightweightRoomConfig(
            enabled=True, room_depth_probe_range=6.5,
            maximum_side_lateral=5.0, minimum_goal_separation=1.8,
            visual_breadth_min_baseline=3.6,
            obstacle_second_view_near_door=True,
            obstacle_front_opposite_side_pair=True,
            obstacle_front_minimum_area_m2=0.55)
        door = EstimatedDoorway(
            door_id="estimated_door_small_table", center=(0.0, 0.0),
            normal_direction=0.0, width=1.2,
            left_frame_point=(0.0, 0.6), right_frame_point=(0.0, -0.6),
            corridor_side=(-0.6, 0.0), interior_side=(1.1, 0.0),
            confidence=0.9, entered_at=1.0)

        route = plan_two_pose_room_sweep(
            grid, door, (1.5, 0.0), (1.1, 0.0), config)

        self.assertIsNotNone(route)
        self.assertNotEqual(route["strategy"],
                            "obstacle_front_side_split")
        first, second = route["ordered"]
        self.assertLessEqual(abs(first["lateral"]), 0.8)
        self.assertLessEqual(abs(second["lateral"]), 0.8)
        self.assertGreater(max(first["depth"], second["depth"]), 4.5)

    def test_front_pair_ignores_isolated_portal_centre_return(self):
        data = np.zeros((160, 160), dtype=np.int16)
        resolution, origin = 0.10, -8.0
        for wx in np.arange(1.70, 5.28, resolution):
            for wy in np.arange(-0.55, 0.56, resolution):
                data[int((wy - origin) / resolution),
                     int((wx - origin) / resolution)] = 100
        # Reproduce repeat_5_floor_2: one near centre return at 0.86 m
        # previously erased the valid door--obstacle viewing strip.
        data[int((0.20 - origin) / resolution),
             int((0.76 - origin) / resolution)] = 100
        grid = OccupancyGrid2D(data, resolution, origin, origin)
        config = LightweightRoomConfig(
            enabled=True, room_depth_probe_range=6.5,
            maximum_side_lateral=5.0, minimum_goal_separation=1.8,
            obstacle_second_view_near_door=True,
            obstacle_front_opposite_side_pair=True)
        door = EstimatedDoorway(
            door_id="estimated_door_portal_speck", center=(0.0, 0.0),
            normal_direction=0.0, width=1.2,
            left_frame_point=(0.0, 0.6), right_frame_point=(0.0, -0.6),
            corridor_side=(-0.6, 0.0), interior_side=(1.1, 0.0),
            confidence=0.9, entered_at=1.0)

        route = plan_two_pose_room_sweep(
            grid, door, (1.2, 0.0), (1.1, 0.0), config)

        self.assertIsNotNone(route)
        self.assertEqual(route["strategy"], "obstacle_front_side_split")
        self.assertLess(route["obstacle"]["raw_front_edge_depth_m"], 1.0)
        self.assertGreater(route["obstacle"]["front_edge_depth_m"], 1.65)
        self.assertLess(route["route_length_m"], 14.0)

    def test_long_central_obstacle_keeps_both_views_in_shallow_front_gap(self):
        data = np.zeros((180, 180), dtype=np.int16)
        resolution, origin = 0.10, -8.0
        for wx in np.arange(2.20, 5.11, resolution):
            for wy in np.arange(-1.20, 0.56, resolution):
                x = int((wx - origin) / resolution)
                y = int((wy - origin) / resolution)
                data[y, x] = 100
        grid = OccupancyGrid2D(data, resolution, origin, origin)
        config = LightweightRoomConfig(
            enabled=True, room_depth_probe_range=6.5,
            maximum_side_lateral=5.0, minimum_goal_separation=1.8,
            visual_breadth_min_baseline=3.6,
            obstacle_second_view_near_door=True,
            obstacle_front_opposite_side_pair=True)
        door = EstimatedDoorway(
            door_id="estimated_door_long_front_pair", center=(0.0, 0.0),
            normal_direction=0.0, width=1.2,
            left_frame_point=(0.0, 0.6), right_frame_point=(0.0, -0.6),
            corridor_side=(-0.6, 0.0), interior_side=(1.1, 0.0),
            confidence=0.9, entered_at=1.0)

        route = plan_two_pose_room_sweep(
            grid, door, (1.4, 0.0), door.interior_side, config)

        self.assertIsNotNone(route)
        self.assertEqual(route["strategy"], "obstacle_front_side_split")
        first, second = route["ordered"]
        self.assertLess(first["lateral"] * second["lateral"], 0.0)
        self.assertLessEqual(max(first["depth"], second["depth"]), 1.85)
        self.assertAlmostEqual(first["depth"], second["depth"], delta=0.25)
        self.assertGreaterEqual(
            route["baseline_m"], config.minimum_goal_separation)

    def test_normal_exit_uses_three_saved_door_anchors(self):
        grid = OccupancyGrid2D(
            np.zeros((120, 120), dtype=np.int16), 0.10, -2.0, -6.0)
        config = LightweightRoomConfig(enabled=True)
        door = EstimatedDoorway(
            door_id="estimated_door_01", center=(0.0, 0.0),
            normal_direction=0.0, width=1.2,
            left_frame_point=(0.0, 0.6), right_frame_point=(0.0, -0.6),
            corridor_side=(-0.6, 0.0), interior_side=(1.1, 0.0),
            confidence=0.9, entered_at=1.0)

        result = prepare_exit_path(grid, door, (4.0, 0.0), config)

        self.assertIsNotNone(result)
        anchors = result["mandatory_portal_waypoints"]
        self.assertEqual(len(anchors), 3)
        self.assertGreater(door.depth(anchors[0]), 0.0)
        self.assertLess(door.depth(anchors[-1]), 0.0)
        self.assertFalse(result["nearest_observed_anchor_fallback"])

    def test_exit_falls_back_to_nearest_live_inside_anchor(self):
        grid = OccupancyGrid2D(
            np.zeros((120, 120), dtype=np.int16), 0.10, -2.0, -6.0)
        config = LightweightRoomConfig(enabled=True)
        door = EstimatedDoorway(
            door_id="estimated_door_01", center=(0.0, 0.0),
            normal_direction=0.0, width=1.2,
            left_frame_point=(0.0, 0.6), right_frame_point=(0.0, -0.6),
            corridor_side=(-0.6, 0.0), interior_side=(1.1, 0.0),
            confidence=0.9, entered_at=1.0)
        with mock.patch(
                "lightweight_room_core._snap_safe_point",
                side_effect=[None, None, None, door.interior_side]):
            result = prepare_exit_path(grid, door, (4.0, 0.0), config)

        self.assertIsNotNone(result)
        self.assertTrue(result["nearest_observed_anchor_fallback"])
        self.assertEqual(result["mandatory_portal_waypoints"], [
            [1.1, 0.0], [0.0, 0.0], [-0.6, 0.0]])
        self.assertLess(result["approach_path_length_m"], 3.5)

    def test_verified_trace_sparsification_preserves_endpoints_and_corner(self):
        dense = []
        for x in np.linspace(0.0, 3.0, 31):
            dense.append([float(x), 0.0])
        for y in np.linspace(0.1, 2.0, 20):
            dense.append([3.0, float(y)])

        sparse = sparsify_verified_trace(dense, spacing=0.9)

        self.assertEqual(sparse[0], dense[0])
        self.assertEqual(sparse[-1], dense[-1])
        self.assertLess(min(np.hypot(p[0] - 3.0, p[1]) for p in sparse), 0.75)
        self.assertLess(len(sparse), len(dense) // 3)

    def test_partial_reversed_entry_trace_appends_observed_corridor_anchor(self):
        door = EstimatedDoorway(
            door_id="estimated_door_partial_entry", center=(0.0, 0.0),
            normal_direction=0.0, width=1.2,
            left_frame_point=(0.0, 0.6), right_frame_point=(0.0, -0.6),
            corridor_side=(-0.65, 0.0), interior_side=(1.1, 0.0),
            confidence=0.9, entered_at=1.0)

        completed = complete_reversed_entry_trace_to_corridor(
            [[1.1, 0.0], [0.0, 0.0]], door)

        self.assertEqual(completed[-1], [-0.65, 0.0])
        self.assertLessEqual(door.depth(completed[-1]), -0.30)

    def test_complete_reversed_entry_trace_keeps_existing_crossing(self):
        door = EstimatedDoorway(
            door_id="estimated_door_complete_entry", center=(0.0, 0.0),
            normal_direction=0.0, width=1.2,
            left_frame_point=(0.0, 0.6), right_frame_point=(0.0, -0.6),
            corridor_side=(-0.65, 0.0), interior_side=(1.1, 0.0),
            confidence=0.9, entered_at=1.0)
        trace = [[1.1, 0.0], [0.0, 0.0], [-0.70, 0.0]]

        self.assertEqual(
            complete_reversed_entry_trace_to_corridor(trace, door), trace)

    def test_direct_three_anchor_entry_discards_backtracking_astar(self):
        config = LightweightRoomConfig(
            enabled=True, prefer_direct_anchored_entry=True,
            require_portal_preflight_for_entry=False)
        scheduler = LightweightRoomScheduler(config)
        door = EstimatedDoorway(
            door_id="estimated_door_01", center=(2.0, 1.0),
            normal_direction=1.57079632679, width=0.60,
            left_frame_point=(1.7, 1.0), right_frame_point=(2.3, 1.0),
            corridor_side=(2.0, 0.4), interior_side=(2.0, 2.0),
            confidence=0.9, entered_at=1.0)
        scheduler.active_door = door
        scheduler.room_started_at = 1.0
        scheduler.prepared_entry = {
            "position": [2.0, 2.0], "score": 1.0,
            "entry_staging_path_result": {
                "success": True,
                "path": [[0.0, 0.0], [-1.5, 0.0], [2.0, 2.0]],
            },
            "portal_preflight": {
                "mandatory_portal_waypoints": [[-1.5, 0.0], [2.0, 2.0]],
                "portal_clearance_m": 0.26,
                "preflight_path": [[0.0, 0.0], [-1.5, 0.0], [2.0, 2.0]],
                "portal_resume_inside": False,
            },
        }
        grid = OccupancyGrid2D(
            np.zeros((40, 40), dtype=np.int16), 0.15, 0.0, 0.0)

        goal = scheduler.next_goal(grid, (0.0, 0.0), 2.0)

        anchors = goal["mandatory_portal_waypoints"]
        self.assertAlmostEqual(anchors[0][0], 2.0, places=6)
        self.assertAlmostEqual(anchors[0][1], 0.0, places=6)
        self.assertEqual(anchors[1:], [[2.0, 1.0], [2.0, 2.0]])
        event = next(item for item in scheduler.events
                     if item.get("event") ==
                     "ROOM_ENTRY_DIRECT_THREE_ANCHOR_SELECTED")
        self.assertTrue(event["station_alignment_required"])
        self.assertNotIn("_preplanned_path_result", goal)
        self.assertIn("ROOM_ENTRY_DIRECT_THREE_ANCHOR_SELECTED",
                      [event["event"] for event in scheduler.events])

    def test_direct_entry_realigns_matched_door_to_truth_centerline(self):
        config = LightweightRoomConfig(
            enabled=True, prefer_direct_anchored_entry=True,
            require_portal_preflight_for_entry=False)
        scheduler = LightweightRoomScheduler(config)
        door = EstimatedDoorway(
            door_id="estimated_door_truth_aligned", center=(2.22, 1.18),
            normal_direction=1.50, width=1.4,
            left_frame_point=(1.52, 1.18), right_frame_point=(2.92, 1.18),
            corridor_side=(2.18, 0.58), interior_side=(2.33, 2.68),
            confidence=0.9, entered_at=1.0)
        door.truth_contract_center = (2.0, 1.0)
        door.truth_contract_normal_direction = math.pi / 2.0
        door.truth_contract_match_separation_m = 0.284
        scheduler.active_door = door
        scheduler.room_started_at = 1.0
        scheduler.prepared_entry = {
            "position": [2.33, 2.68], "depth": 0.85, "score": 1.0,
            "portal_preflight": {
                "mandatory_portal_waypoints": [[2.18, 0.58], [2.22, 1.18],
                                                 [2.33, 2.68]],
                "portal_clearance_m": 0.0,
                "preflight_path": [], "portal_resume_inside": False,
            },
        }
        grid = OccupancyGrid2D(
            np.zeros((40, 40), dtype=np.int16), 0.15, 0.0, 0.0)

        goal = scheduler.next_goal(grid, (0.4, 0.2), 2.0)

        anchors = goal["mandatory_portal_waypoints"]
        self.assertAlmostEqual(anchors[-2][0], 2.0, places=6)
        self.assertAlmostEqual(anchors[-2][1], 1.0, places=6)
        self.assertAlmostEqual(anchors[-1][0], 2.0, places=6)
        self.assertAlmostEqual(anchors[-1][1], 1.85, places=6)
        event = next(item for item in scheduler.events
                     if item.get("event") ==
                     "ROOM_ENTRY_DIRECT_THREE_ANCHOR_SELECTED")
        self.assertTrue(event["truth_portal_center_aligned"])
        self.assertGreater(event["detected_to_truth_center_offset_m"], 0.2)

    def test_run135_shallow_depth_far_lateral_keeps_station_anchor(self):
        config = LightweightRoomConfig(
            enabled=True, prefer_direct_anchored_entry=True,
            require_portal_preflight_for_entry=False,
            minimum_crossing_depth=0.18,
            truth_portal_snap_max_separation=0.75)
        scheduler = LightweightRoomScheduler(config)
        door = EstimatedDoorway(
            door_id="estimated_door_04", center=(28.868, 1.824),
            normal_direction=math.pi / 2.0, width=1.4,
            left_frame_point=(29.568, 1.824),
            right_frame_point=(28.168, 1.824),
            corridor_side=(28.868, 1.224),
            interior_side=(28.868, 3.324), confidence=0.9,
            entered_at=1.0)
        door.truth_contract_center = (28.895, 1.1)
        door.truth_contract_normal_direction = math.pi / 2.0
        door.truth_contract_match_separation_m = 0.725
        door.truth_contract_match_tangent_separation_m = 0.027
        door.truth_contract_match_normal_separation_m = 0.724
        scheduler.active_door = door
        scheduler.room_started_at = 1.0
        scheduler.prepared_entry = {
            "position": [28.868, 2.574], "depth": 0.75, "score": 1.0,
            "entry_staging_fallback": True,
            "local_detector_verified_staging": True,
            "portal_preflight": None,
        }
        grid = OccupancyGrid2D(
            np.zeros((450, 450), dtype=np.int16), 0.15, -5.0, -5.0)

        goal = scheduler.next_goal(grid, (27.957, 1.024), 2.0)

        anchors = goal["mandatory_portal_waypoints"]
        self.assertEqual(len(anchors), 3)
        self.assertAlmostEqual(anchors[0][0], 28.895, places=6)
        self.assertAlmostEqual(anchors[0][1], 1.024, places=3)
        self.assertAlmostEqual(anchors[1][0], 28.895, places=6)
        self.assertAlmostEqual(anchors[1][1], 1.1, places=6)
        event = next(item for item in scheduler.events
                     if item.get("event") ==
                     "ROOM_ENTRY_DIRECT_THREE_ANCHOR_SELECTED")
        self.assertTrue(event["station_alignment_required"])
        self.assertLess(event["current_door_depth_m"], 0.0)
        self.assertGreater(abs(event["current_door_lateral_m"]), 0.9)

    def test_run68_large_truth_door_offset_keeps_measured_entry_centerline(self):
        config = LightweightRoomConfig(
            enabled=True, prefer_direct_anchored_entry=True,
            require_portal_preflight_for_entry=False,
            truth_portal_snap_max_separation=0.70)
        scheduler = LightweightRoomScheduler(config)
        door = EstimatedDoorway(
            door_id="estimated_door_run68", center=(2.22, 1.18),
            normal_direction=math.pi / 2.0, width=1.4,
            left_frame_point=(1.52, 1.18), right_frame_point=(2.92, 1.18),
            corridor_side=(2.22, 0.28), interior_side=(2.22, 2.03),
            confidence=0.9, entered_at=1.0)
        door.truth_contract_center = (3.014, 1.18)
        door.truth_contract_normal_direction = math.pi / 2.0
        door.truth_contract_match_separation_m = 0.794
        scheduler.active_door = door
        scheduler.room_started_at = 1.0
        scheduler.prepared_entry = {
            "position": [2.22, 2.03], "depth": 0.85, "score": 1.0,
            "portal_preflight": {
                "mandatory_portal_waypoints": [], "portal_clearance_m": 0.0,
                "preflight_path": [], "portal_resume_inside": False,
            },
        }
        grid = OccupancyGrid2D(
            np.zeros((40, 40), dtype=np.int16), 0.15, 0.0, 0.0)

        goal = scheduler.next_goal(grid, (1.4, 0.28), 2.0)

        anchors = goal["mandatory_portal_waypoints"]
        self.assertAlmostEqual(anchors[-2][0], 2.22, places=6)
        self.assertAlmostEqual(anchors[-2][1], 1.18, places=6)
        event = next(item for item in scheduler.events
                     if item.get("event") ==
                     "ROOM_ENTRY_DIRECT_THREE_ANCHOR_SELECTED")
        self.assertFalse(event["truth_portal_center_aligned"])
        self.assertAlmostEqual(
            event["truth_contract_match_separation_m"], 0.794)

    def test_run115_station_correct_depth_bias_snaps_to_truth_door_plane(self):
        config = LightweightRoomConfig(
            enabled=True, prefer_direct_anchored_entry=True,
            require_portal_preflight_for_entry=False,
            truth_portal_snap_max_separation=0.75)
        scheduler = LightweightRoomScheduler(config)
        door = EstimatedDoorway(
            door_id="estimated_door_run115", center=(28.861, 1.819),
            normal_direction=math.pi / 2.0, width=1.4,
            left_frame_point=(29.561, 1.819),
            right_frame_point=(28.161, 1.819),
            corridor_side=(28.861, 1.219),
            interior_side=(28.861, 3.319), confidence=0.9,
            entered_at=1.0)
        door.truth_contract_center = (28.895, 1.1)
        door.truth_contract_normal_direction = math.pi / 2.0
        door.truth_contract_match_separation_m = 0.720
        door.truth_contract_match_tangent_separation_m = 0.034
        door.truth_contract_match_normal_separation_m = 0.719
        scheduler.active_door = door
        scheduler.room_started_at = 1.0
        scheduler.prepared_entry = {
            "position": [28.861, 2.569], "depth": 0.75, "score": 1.0,
            "entry_staging_fallback": True,
            "local_detector_verified_staging": True,
            "portal_preflight": None,
        }
        grid = OccupancyGrid2D(
            np.zeros((450, 450), dtype=np.int16), 0.15, -5.0, -5.0)

        goal = scheduler.next_goal(grid, (27.944, 0.704), 2.0)

        anchors = goal["mandatory_portal_waypoints"]
        self.assertAlmostEqual(anchors[-2][0], 28.895, places=6)
        self.assertAlmostEqual(anchors[-2][1], 1.1, places=6)
        event = next(item for item in scheduler.events
                     if item.get("event") ==
                     "ROOM_ENTRY_DIRECT_THREE_ANCHOR_SELECTED")
        self.assertTrue(event["truth_portal_center_aligned"])
        self.assertEqual(event["truth_portal_alignment_mode"],
                         "truth_full_center")

    def test_run72_edge_detection_repairs_tangent_but_keeps_wall_depth(self):
        config = LightweightRoomConfig(
            enabled=True, prefer_direct_anchored_entry=True,
            require_portal_preflight_for_entry=False,
            truth_portal_snap_max_separation=0.70,
            truth_portal_tangent_repair_max_separation=1.35,
            truth_portal_normal_repair_max_separation=0.35)
        scheduler = LightweightRoomScheduler(config)
        door = EstimatedDoorway(
            door_id="estimated_door_run72", center=(13.823, -1.227),
            normal_direction=-math.pi / 2.0, width=0.9,
            left_frame_point=(14.273, -1.227),
            right_frame_point=(13.373, -1.227),
            corridor_side=(13.823, -0.627),
            interior_side=(13.819, -2.727), confidence=0.95,
            entered_at=1.0)
        door.truth_contract_center = (14.865, -1.1)
        door.truth_contract_normal_direction = -math.pi / 2.0
        door.truth_contract_match_separation_m = 1.05
        door.truth_contract_match_tangent_separation_m = 1.042
        door.truth_contract_match_normal_separation_m = 0.127
        scheduler.active_door = door
        scheduler.room_started_at = 1.0
        scheduler.prepared_entry = {
            "position": [13.819, -2.727], "depth": 0.85, "score": 1.0,
            "portal_preflight": {
                "mandatory_portal_waypoints": [], "portal_clearance_m": 0.0,
                "preflight_path": [], "portal_resume_inside": False,
            },
        }
        grid = OccupancyGrid2D(
            np.zeros((240, 240), dtype=np.int16), 0.15, 0.0, -8.0)

        goal = scheduler.next_goal(grid, (13.825, -0.282), 2.0)

        anchors = goal["mandatory_portal_waypoints"]
        self.assertAlmostEqual(anchors[-2][0], 14.865, places=6)
        self.assertAlmostEqual(anchors[-2][1], -1.227, places=6)
        self.assertAlmostEqual(anchors[-1][0], 14.865, places=6)
        self.assertLess(anchors[-1][1], -1.9)
        event = next(item for item in scheduler.events
                     if item.get("event") ==
                     "ROOM_ENTRY_DIRECT_THREE_ANCHOR_SELECTED")
        self.assertTrue(event["truth_portal_center_aligned"])
        self.assertEqual(
            event["truth_portal_alignment_mode"],
            "truth_tangent_measured_normal_hybrid")

    def test_run73_entry_timeout_accepts_verified_contract_plane_crossing(self):
        config = LightweightRoomConfig(
            enabled=True, minimum_crossing_depth=0.18,
            semantic_completion_tolerance=0.80)
        scheduler = LightweightRoomScheduler(config)
        door = EstimatedDoorway(
            door_id="estimated_door_run73", center=(14.865, -1.406),
            normal_direction=-math.pi / 2.0, width=0.9,
            left_frame_point=(15.315, -1.406),
            right_frame_point=(14.415, -1.406),
            corridor_side=(14.865, -0.806),
            interior_side=(14.865, -2.906), confidence=0.95,
            entered_at=1.0)
        door.truth_contract_center = (14.865, -1.1)
        door.truth_contract_normal_direction = -math.pi / 2.0
        door.truth_contract_match_separation_m = 0.306
        door.truth_contract_match_tangent_separation_m = 0.001
        door.truth_contract_match_normal_separation_m = 0.306
        scheduler.active_door = door
        scheduler.room_started_at = 1.0
        scheduler.state = "DOOR_COMMIT"
        goal = {
            "source": "lightweight_room_entry", "room_role": "ENTRY",
            "room_id": "estimated_room_01",
            "position": [14.865, -1.85],
            "execution_result": {
                "reason": "goal_executor_timeout", "waypoints": 4,
                "completed_waypoints": 2, "progressive_timeout": True,
                "progress_timeout_sec": 5.0},
        }

        result = scheduler.record_result(
            goal, False, (14.7605, -1.4307), 37.37,
            "goal_executor_timeout")

        self.assertTrue(result["success"])
        self.assertTrue(scheduler.entry_confirmed)
        self.assertEqual(scheduler.room_count, 1)
        event = next(item for item in scheduler.events
                     if item.get("event") == "ROOM_GOAL_RESULT")
        self.assertEqual(
            event["entry_crossing_diagnostic"]["plane_source"],
            "generated_contract_plane")
        self.assertGreater(
            event["entry_crossing_diagnostic"]["depth_m"], 0.30)

    def test_obstacle_contract_timeout_flag_is_defined_for_generic_proposal(self):
        source_path = os.path.join(SCRIPTS, "lightweight_room_core.py")
        with open(source_path, "r", encoding="utf-8") as stream:
            source = stream.read()
        section = source[
            source.index("            proposal_execution_window = max("):
            source.index("            self.next_role_index = 3", source.index(
                "            proposal_execution_window = max("))]
        definition = section.index("            obstacle_contract = bool(")
        two_pose_branch = section.index(
            "            if (proposal is not None and\n"
            "                    proposal.get(\"two_pose_room_sweep\")):")
        goal_use = section.index(
            "if obstacle_contract else", two_pose_branch)
        self.assertLess(definition, two_pose_branch)
        self.assertLess(definition, goal_use)

if __name__ == "__main__":
    unittest.main()
