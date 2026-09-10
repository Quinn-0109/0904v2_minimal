#!/usr/bin/env python3
"""Tests for receding-horizon observed-free exploration paths."""

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
from local_exploration_path_core import (
    CandidatePath, LocalPathConfig, build_candidate_path,
    cell_is_free_with_clearance, generate_candidate_paths,
    path_commit_released, path_score)
from planner_interface import AStarPlanner


def local_frontier_grid():
    data = np.full((120, 120), -1, dtype=np.int16)
    data[40:80, 40:100] = 0
    return OccupancyGrid2D(data, 0.1, -6.0, -6.0)


class LocalExplorationPathTest(unittest.TestCase):
    def setUp(self):
        self.grid = local_frontier_grid()
        self.pose = (0.0, 0.0, 0.0)
        self.planner = AStarPlanner(
            clearance=0.25, reached_tolerance=0.10,
            allow_blocked_start=False)
        self.config = LocalPathConfig(
            waypoint_spacing=0.75, waypoint_count=4,
            minimum_waypoints=2, free_clearance=0.25,
            sensor_range=2.0)

    def test_generated_waypoints_are_observed_free_and_reachable(self):
        selected, candidates = generate_candidate_paths(
            self.grid, self.pose, 0.0, self.planner, [], [],
            self.config,
            scan_checker=lambda point, yaw: (True, "scan_safe", 0.0))
        self.assertIsNotNone(selected)
        self.assertGreaterEqual(len(selected.waypoints), 2)
        for waypoint in selected.waypoints:
            cell = self.grid.world_to_cell((waypoint.x, waypoint.y))
            self.assertEqual(int(self.grid.data[cell[1], cell[0]]), 0)
            self.assertTrue(cell_is_free_with_clearance(
                self.grid, cell, self.config.free_clearance))
            self.assertEqual(waypoint.safety_status, "scan_safe")
        self.assertTrue(any(not item.rejected for item in candidates))

    def test_information_gain_is_unique_across_waypoints(self):
        candidate = build_candidate_path(
            self.grid, self.pose, 0.0, 0.0, self.planner, [],
            self.config, "gain")
        self.assertGreater(candidate.information_cell_count, 0)
        self.assertEqual(
            sum(item.unknown_cell_count for item in candidate.waypoints),
            candidate.information_cell_count)
        self.assertAlmostEqual(
            candidate.total_information_gain,
            candidate.information_cell_count * self.grid.resolution ** 2)

    def test_scan_unsafe_waypoints_are_deleted(self):
        candidate = build_candidate_path(
            self.grid, self.pose, 0.0, 0.0, self.planner, [],
            self.config, "unsafe",
            scan_checker=lambda point, yaw: (
                point[0] <= 1.0, "scan_safe" if point[0] <= 1.0
                else "scan_collision", 0.0))
        self.assertTrue(all(item.x <= 1.0 for item in candidate.waypoints))
        self.assertTrue(candidate.rejected)

    def test_backward_and_revisit_cost_are_penalized(self):
        backward = build_candidate_path(
            self.grid, self.pose, 0.0, math.pi, self.planner, [],
            self.config, "backward")
        self.assertLess(backward.forward_progress, 0.0)
        self.assertGreaterEqual(backward.revisit_cost,
                                self.config.backward_penalty)
        visited = [(0.75, 0.0, 0.0), (1.5, 0.0, 0.0)]
        revisited = build_candidate_path(
            self.grid, self.pose, 0.0, 0.0, self.planner, visited,
            self.config, "visited")
        novel = build_candidate_path(
            self.grid, self.pose, 0.0, 0.0, self.planner, [],
            self.config, "novel")
        self.assertGreater(revisited.revisit_cost, novel.revisit_cost)

    def test_score_uses_requested_weights(self):
        config = LocalPathConfig(
            alpha=1.0, beta=0.4, gamma=1.0,
            delta=0.5, eta=1.0, mu=1.5)
        candidate = CandidatePath(
            "score", 0.0, "test", None,
            total_information_gain=10.0, path_length=2.0,
            revisit_cost=1.0, turning_cost=0.4,
            risk_cost=0.5, forward_progress=2.0)
        self.assertAlmostEqual(path_score(candidate, config), 10.5)

    def test_commit_release_uses_or_conditions(self):
        self.assertFalse(path_commit_released(2, 1.4, 11.9, 2.0))
        self.assertTrue(path_commit_released(3, 0.0, 0.0, 2.0))
        self.assertTrue(path_commit_released(0, 1.5, 0.0, 2.0))
        self.assertTrue(path_commit_released(0, 0.0, 12.0, 2.0))
        self.assertTrue(path_commit_released(0, 0.0, 0.0, 0.5))


if __name__ == "__main__":
    unittest.main()
