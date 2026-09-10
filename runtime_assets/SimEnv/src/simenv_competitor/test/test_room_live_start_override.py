#!/usr/bin/env python3
import math
import os
import sys
import unittest


SCRIPT_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'scripts'))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from baseline_planning_core import (
    room_live_start_override_allowed,
    truth_open_room_goal_override_allowed,
)


class RoomLiveStartOverrideTest(unittest.TestCase):
    def setUp(self):
        self.current = (2.10, 14.86, 5.58, 0.0)
        self.trajectory = [
            (0.0, 1.95, 14.86, 5.58),
            (0.1, 2.02, 14.86, 5.58),
            (0.2, 2.10, 14.86, 5.58),
        ]

    def test_truth_crossed_continuous_active_room_is_allowed(self):
        self.assertTrue(room_live_start_override_allowed(
            self.current, self.trajectory, True, True, True))

    def test_requires_truth_crossing_active_room_and_authorization(self):
        for crossing, active, truth in (
                (False, True, True), (True, False, True),
                (True, True, False)):
            self.assertFalse(room_live_start_override_allowed(
                self.current, self.trajectory, crossing, active, truth))

    def test_rejects_reset_or_teleport_discontinuity(self):
        discontinuous = list(self.trajectory)
        discontinuous.insert(1, (0.05, 8.0, 20.0, 5.58))
        self.assertFalse(room_live_start_override_allowed(
            self.current, discontinuous, True, True, True))

    def test_rejects_stale_or_nonfinite_last_pose(self):
        stale = list(self.trajectory)
        stale[-1] = (0.2, 2.30, 14.86, 5.58)
        self.assertFalse(room_live_start_override_allowed(
            self.current, stale, True, True, True))
        invalid = list(self.trajectory)
        invalid[-1] = (0.2, math.nan, 14.86, 5.58)
        self.assertFalse(room_live_start_override_allowed(
            self.current, invalid, True, True, True))

    def test_truth_open_endpoint_override_is_narrow(self):
        diagnostic = {
            "truth_layout_centerline_fallback": True,
            "live_3d_audit_required": True,
            "two_pose_strategy":
                "truth_layout_open_centerline_deep_near",
            "two_pose_obstacle": None,
        }
        self.assertTrue(truth_open_room_goal_override_allowed(
            "lightweight_room_semantic", "G3", diagnostic, True, True))
        for changed in (
                {"two_pose_strategy":
                 "obstacle_front_opposite_sides"},
                {"two_pose_obstacle": {"id": "table"}},
                {"live_3d_audit_required": False}):
            rejected = dict(diagnostic)
            rejected.update(changed)
            self.assertFalse(truth_open_room_goal_override_allowed(
                "lightweight_room_semantic", "G3", rejected, True, True))
        self.assertFalse(truth_open_room_goal_override_allowed(
            "lightweight_room_semantic", "ENTRY", diagnostic, True, True))
        self.assertFalse(truth_open_room_goal_override_allowed(
            "corridor_sweep", "G3", diagnostic, True, True))
        self.assertFalse(truth_open_room_goal_override_allowed(
            "lightweight_room_semantic", "G3", diagnostic, False, True))


if __name__ == '__main__':
    unittest.main()
