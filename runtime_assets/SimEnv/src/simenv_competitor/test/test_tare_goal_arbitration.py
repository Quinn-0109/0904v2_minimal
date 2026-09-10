#!/usr/bin/env python3
"""Regression tests for Official TARE rolling-goal arbitration."""

import math
import os
import sys
import unittest

import numpy as np

SCRIPT_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "scripts"))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from baseline_planning_core import OccupancyGrid2D
from tare_goal_arbitration_core import (
    ArbitrationConfig, GoalMetrics, TareGoalArbitrationPolicy,
    goal_score, map_goal_metrics)


def frontier_rich_grid():
    data = np.full((200, 200), -1, dtype=np.int16)
    data[80:120, 80:120] = 0
    return OccupancyGrid2D(data, 0.1, -10.0, -10.0)


class TareGoalArbitrationTest(unittest.TestCase):
    def setUp(self):
        self.grid = frontier_rich_grid()
        self.pose = (0.0, 0.0, 0.0)

    def test_minimum_commitment_blocks_rolling_update(self):
        policy = TareGoalArbitrationPolicy()
        self.assertTrue(policy.decide(
            (1.0, 0.0, 0.0), self.pose, 0.0, self.grid, 10.0).accepted)
        decision = policy.decide(
            (1.0, 1.0, 0.0), self.pose, 0.0, self.grid, 15.0)
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, "min_goal_commit_time")

    def test_preemption_requires_explicit_score_margin(self):
        policy = TareGoalArbitrationPolicy()
        policy.decide((1.0, 0.0, 0.0), self.pose, 0.0,
                      self.grid, 10.0)
        decision = policy.decide(
            (1.0, 0.6, 0.0), self.pose, 0.0, self.grid, 23.0)
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, "insufficient_score_gain")

    def test_backward_goal_gets_five_point_penalty(self):
        metrics = GoalMetrics(unknown_area=10.0, frontier_cells=20)
        forward, forward_backward = goal_score(
            self.pose, 0.0, (1.0, 0.0, 0.0), metrics, 5.0)
        backward, is_backward = goal_score(
            self.pose, 0.0, (-1.0, 0.0, 0.0), metrics, 5.0)
        self.assertFalse(forward_backward)
        self.assertTrue(is_backward)
        self.assertAlmostEqual(forward - backward, 5.0)

    def test_active_region_rejects_jump_to_unrelated_region(self):
        policy = TareGoalArbitrationPolicy()
        policy.decide((1.0, 0.0, 0.0), self.pose, 0.0,
                      self.grid, 10.0)
        decision = policy.decide(
            (6.0, 0.0, 0.0), self.pose, 0.0, self.grid, 23.0)
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, "active_region_commitment")

    def test_path_failure_releases_commitment(self):
        policy = TareGoalArbitrationPolicy()
        policy.decide((1.0, 0.0, 0.0), self.pose, 0.0,
                      self.grid, 10.0)
        policy.mark_failure()
        decision = policy.decide(
            (-2.0, 0.0, 0.0), self.pose, 0.0, self.grid, 11.0)
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.reason, "current_goal_failed")

    def test_repeated_failure_invalidates_and_blacklists_goal(self):
        config = ArbitrationConfig(
            maximum_failures=3, failed_goal_blacklist_duration=20.0)
        policy = TareGoalArbitrationPolicy(config)
        failed = (1.0, 0.0, 0.0)
        policy.decide(failed, self.pose, 0.0, self.grid, 1.0)
        self.assertIsNone(policy.mark_failure(2.0))
        self.assertIsNone(policy.mark_failure(3.0))
        self.assertEqual(policy.mark_failure(4.0), failed)
        self.assertIsNone(policy.current_goal)

        held = policy.decide(failed, self.pose, 0.0, self.grid, 5.0)
        self.assertFalse(held.accepted)
        self.assertEqual(held.reason, "failed_goal_blacklisted")
        alternate = policy.decide(
            (1.0, 1.0, 0.0), self.pose, 0.0, self.grid, 6.0)
        self.assertTrue(alternate.accepted)

    def test_supersede_burst_enters_forced_commit(self):
        config = ArbitrationConfig(
            min_goal_commit_time=0.0, max_supersede_per_minute=1)
        policy = TareGoalArbitrationPolicy(config)
        policy.decide((1.0, 0.0, 0.0), self.pose, 0.0,
                      self.grid, 1.0)
        policy.mark_failure()
        policy.decide((2.0, 0.0, 0.0), self.pose, 0.0,
                      self.grid, 2.0)
        policy.mark_failure()
        second = policy.decide((3.0, 0.0, 0.0), self.pose, 0.0,
                               self.grid, 3.0)
        self.assertTrue(second.forced_commit_mode)
        held = policy.decide((3.0, 1.0, 0.0), self.pose, 0.0,
                             self.grid, 4.0)
        self.assertFalse(held.accepted)
        self.assertEqual(held.reason, "forced_commit_mode")

    def test_map_metrics_measure_unknown_and_frontier(self):
        metrics = map_goal_metrics(self.grid, self.pose, 4.0)
        self.assertGreater(metrics.unknown_area, 3.0)
        self.assertGreaterEqual(metrics.frontier_cells, 8)


if __name__ == "__main__":
    unittest.main()
