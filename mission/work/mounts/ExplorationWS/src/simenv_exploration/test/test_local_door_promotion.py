#!/usr/bin/env python3

import os
import sys
import unittest
from unittest import mock

import numpy as np

SCRIPTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, SCRIPTS)

from baseline_planning_core import OccupancyGrid2D  # noqa: E402
from lightweight_room_core import (  # noqa: E402
    EstimatedDoorway, LightweightRoomConfig, LightweightRoomScheduler,
    prepare_exit_path,
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

    def test_confirmed_local_candidate_enters_common_preflight(self):
        sentinel = object()
        self.scheduler._prepare_and_activate = mock.Mock(return_value=sentinel)
        result = self.scheduler.consider_local_door_candidate(
            object(), (2.0, 0.0), self.evidence, 12.0)
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


if __name__ == "__main__":
    unittest.main()
