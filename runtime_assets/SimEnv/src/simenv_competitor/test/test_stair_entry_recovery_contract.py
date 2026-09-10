#!/usr/bin/env python3
"""Regression checks for bounded stair-entry recovery and status truthfulness."""

import inspect
import math
import os
import sys
import unittest
from types import SimpleNamespace


SCRIPT_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "scripts"))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from baseline_telemetry import BaselineTelemetry
from stair_transition_manager import StairTransition


class StairEntryRecoveryContractTest(unittest.TestCase):
    @staticmethod
    def _hot_manager():
        manager = StairTransition.__new__(StairTransition)
        manager.source_floor_index = 0
        manager.truth_pose = [-4.02, 1.75, .31, math.pi / 2.0]
        manager.truth_entry_target = [-4.02, 1.75]
        manager.truth_stair_heading = math.pi / 2.0
        manager.truth_post_policy_restage_tolerance = .12
        manager.truth_model_pose = SimpleNamespace(
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0))
        manager.truth_model_twist = SimpleNamespace(
            linear=SimpleNamespace(x=.01, y=.01, z=.01),
            angular=SimpleNamespace(x=.01, y=.01, z=.01))
        manager.locomotion_ready = True
        manager.locomotion_ever_ready = True
        return manager

    def test_upright_quiet_pre_riser_qualifies_for_hot_handoff(self):
        self.assertTrue(
            self._hot_manager()._truth_pre_riser_hot_handoff_ready())

    def test_low_or_tilted_pre_riser_cannot_enter_fixed_stand(self):
        low = self._hot_manager()
        low.truth_pose[2] = .15
        self.assertFalse(low._truth_pre_riser_posture_upright())
        tilted = self._hot_manager()
        tilted.truth_model_pose.orientation.x = math.sin(.20)
        tilted.truth_model_pose.orientation.w = math.cos(.20)
        self.assertFalse(tilted._truth_pre_riser_posture_upright())

    def test_motion_or_lost_locomotion_blocks_hot_handoff(self):
        moving = self._hot_manager()
        moving.truth_model_twist.linear.x = .20
        self.assertFalse(moving._truth_pre_riser_hot_handoff_ready())
        not_ready = self._hot_manager()
        not_ready.locomotion_ready = False
        self.assertTrue(not_ready._truth_pre_riser_posture_upright())
        self.assertFalse(not_ready._truth_pre_riser_hot_handoff_ready())

    def test_policy_recovery_hold_does_not_reload_policy_each_tick(self):
        # ``tick`` is now only the non-reentrant lock wrapper; state-machine
        # ownership, including the bounded recovery hold, lives in
        # ``_tick_impl``.
        source = inspect.getsource(StairTransition._tick_impl)
        start = source.index("self.truth_entry_policy_recovery_until")
        end = source.index("self.truth_fixed_two_flight_profile", start)
        recovery_hold = source[start:end]
        self.assertIn("self.cmd.publish(Twist())", recovery_hold)
        self.assertIn("self.hold_rl()", recovery_hold)
        self.assertNotIn("self.pub.publish", recovery_hold)

    def test_not_reached_is_never_reported_as_success(self):
        description, colour = BaselineTelemetry._task_description(
            "F1_TO_F2", "STAIR_ENTRY_NOT_REACHED", "", False)
        self.assertIn("上楼失败", description)
        self.assertEqual(colour, "red")

    def test_real_handoff_complete_is_reported_as_success(self):
        description, colour = BaselineTelemetry._task_description(
            "F1_TO_F2", "STAIR_HANDOFF_COMPLETE", "", False)
        self.assertIn("上楼成功", description)
        self.assertEqual(colour, "green")


if __name__ == "__main__":
    unittest.main()
