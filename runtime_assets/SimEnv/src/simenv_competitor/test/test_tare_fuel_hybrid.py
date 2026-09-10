#!/usr/bin/env python3
"""Regression tests for persistent coverage and topology arbitration."""

import math
import os
import sys
import unittest

import numpy as np

SCRIPT_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "scripts"))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from structured_topology_core import (
    CorridorDetector, GeometryParameters, OccupancyMap,
    TemporalTopologyTracker)
from tare_fuel_hybrid_core import (
    FrontierIntent, PersistentCoverageMemory, bilateral_corridor_evidence,
    corridor_entry_ready, forward_candidate_points, intent_key,
    point_in_corridor, pop_next_eligible, rank_fuel_candidates)
from tare_fuel_hybrid_core import select_safe_forward_candidate
from baseline_planning_core import OccupancyGrid2D
from planner_interface import AStarPlanner


class TareFuelHybridTest(unittest.TestCase):
    def test_safe_forward_candidates_prefer_body_heading(self):
        candidates = forward_candidate_points(
            (1.0, 2.0, 0.0), math.pi / 2.0,
            distances=(1.5,), offsets=(0.0, math.radians(15.0)))
        self.assertAlmostEqual(candidates[0][0], 1.0, places=6)
        self.assertAlmostEqual(candidates[0][1], 3.5, places=6)
        self.assertGreater(candidates[1][0], candidates[0][0] - 1.0)

    def test_safe_forward_requires_observed_free_astar_path(self):
        free = OccupancyGrid2D(
            np.zeros((100, 100), dtype=np.int16), 0.1, -5.0, -5.0)
        planner = AStarPlanner(clearance=0.20, reached_tolerance=0.10)
        selected = select_safe_forward_candidate(
            (0.0, 0.0, 0.0), 0.0, planner, free)
        self.assertIsNotNone(selected)
        self.assertAlmostEqual(selected[0], 1.5, places=6)
        self.assertAlmostEqual(selected[1], 0.0, places=6)

        blocked_data = np.zeros((100, 100), dtype=np.int16)
        blocked_data[:, 55:58] = 100
        blocked = OccupancyGrid2D(blocked_data, 0.1, -5.0, -5.0)
        self.assertIsNone(select_safe_forward_candidate(
            (0.0, 0.0, 0.0), 0.0, planner, blocked))

    def test_same_low_gain_frontier_intent_is_rejected(self):
        memory = PersistentCoverageMemory()
        memory.add_intent(FrontierIntent(
            (1.0, 1.0, 0.0), (3.0, 1.0), gain=100.0,
            completed=True))
        repeated = FrontierIntent(
            (1.2, 1.1, 0.0), (3.2, 1.1), gain=20.0)
        self.assertEqual(
            memory.repeat_reason(repeated, maximum_gain=100.0),
            "repeated_frontier_intent")

    def test_same_endpoint_opposite_frontier_remains_eligible(self):
        memory = PersistentCoverageMemory()
        memory.add_pose((1.0, 1.0, 0.0))
        memory.add_intent(FrontierIntent(
            (1.0, 1.0, 0.0), (3.0, 1.0), gain=100.0,
            completed=True))
        opposite = FrontierIntent(
            (1.1, 1.0, 0.0), (-2.0, 1.0), gain=10.0)
        self.assertIsNone(memory.repeat_reason(
            opposite, maximum_gain=100.0,
            strict_visited_endpoint=True))

    def test_covered_transit_does_not_block_novel_endpoint(self):
        memory = PersistentCoverageMemory()
        for x in np.arange(0.0, 2.1, 0.5):
            memory.add_pose((float(x), 0.0, 0.0))
        novel = FrontierIntent((3.2, 0.0, 0.0), (5.0, 0.0), gain=5.0)
        self.assertIsNone(memory.repeat_reason(
            novel, maximum_gain=100.0,
            strict_visited_endpoint=True))

    def test_persistent_memory_survives_arbitration_mode_cycles(self):
        memory = PersistentCoverageMemory()
        completed = FrontierIntent(
            (1.0, 1.0, 0.0), (3.0, 1.0), gain=100.0,
            source="fuel", completed=True)
        memory.add_intent(completed)
        # Mode changes operate on the same task-wide memory object.  Ranking
        # unrelated TARE/FUEL batches must not consume or clear old intents.
        rank_fuel_candidates(
            [{"id": 4, "position": [4.0, 0.0, 0.0],
              "frontier_center": [6.0, 0.0], "reachable": True}],
            [{"candidate_id": 4, "score": 50.0,
              "information_gain": 50.0}],
            (2.0, 0.0, 0.0), memory)
        self.assertIs(memory.intents[0], completed)
        self.assertEqual(memory.repeat_reason(
            FrontierIntent((1.1, 1.0, 0.0), (3.1, 1.0), gain=10.0),
            maximum_gain=100.0), "repeated_frontier_intent")

    def test_old_lidar_coverage_rejects_non_topological_endpoint(self):
        memory = PersistentCoverageMemory(observed_cell_size=0.5)
        memory.add_pose((0.0, 0.0, 0.0))
        memory.observe_free_cells(
            [(x * 0.5, y * 0.5) for x in range(-4, 5)
             for y in range(-4, 5)])
        for x in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5,
                  4.0, 4.5, 5.0, 5.5):
            memory.add_pose((x, 0.0, 0.0))
        candidates = [
            {"id": 1, "position": [0.0, 1.0, 0.0],
             "frontier_center": [0.0, 8.0], "reachable": True,
             "topology_bonus": 0.0},
            {"id": 2, "position": [0.0, -1.0, 0.0],
             "frontier_center": [0.0, -8.0], "reachable": True,
             "topology_bonus": 1.0},
        ]
        scores = [
            {"candidate_id": 1, "score": 100.0, "information_gain": 100.0},
            {"candidate_id": 2, "score": 90.0, "information_gain": 90.0},
        ]
        ranked = rank_fuel_candidates(
            candidates, scores, (5.5, 0.0, 0.0), memory,
            minimum_distance=0.0, maximum_distance=8.0)
        self.assertEqual([item["candidate"]["id"] for item in ranked], [2])

    def test_bootstrap_replay_rejects_old_lobby_execution_point(self):
        memory = PersistentCoverageMemory()
        memory.add_pose((2.325, 0.225, 0.0))
        candidates = [
            {"id": 36, "position": [2.325, 0.225, 0.0],
             "frontier_center": [21.922826, -1.301087],
             "reachable": True, "topology_bonus": 0.0},
            {"id": 40, "position": [0.0, 2.5, 0.0],
             "frontier_center": [0.2, 8.0],
             "reachable": True, "topology_bonus": 1.0},
        ]
        scores = [
            {"candidate_id": 36, "score": 3099.4,
             "information_gain": 6957.0, "local_unknown_volume": 43.3},
            {"candidate_id": 40, "score": 2200.0,
             "information_gain": 4000.0, "local_unknown_volume": 25.0},
        ]
        ranked = rank_fuel_candidates(
            candidates, scores, (0.0, 0.0, 0.0), memory,
            strict_visited_endpoint=True)
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0]["candidate"]["id"], 40)

    def test_invalid_distance_uses_next_candidate(self):
        memory = PersistentCoverageMemory()
        candidates = [
            {"id": 1, "position": [0.1, 0.0, 0.0], "reachable": True},
            {"id": 2, "position": [2.0, 0.0, 0.0], "reachable": True},
        ]
        scores = [
            {"candidate_id": 1, "score": 100.0, "information_gain": 100.0},
            {"candidate_id": 2, "score": 50.0, "information_gain": 50.0},
        ]
        ranked = rank_fuel_candidates(
            candidates, scores, (0.0, 0.0, 0.0), memory)
        self.assertEqual([item["candidate"]["id"] for item in ranked], [2])

    def test_scan_rejection_blacklist_selects_next_candidate(self):
        first = {"intent": FrontierIntent(
            (1.0, 0.0, 0.0), frontier=(2.0, 0.0))}
        second = {"intent": FrontierIntent(
            (0.0, 1.0, 0.0), frontier=(0.0, 2.0))}
        queue = [first, second]
        selected = pop_next_eligible(
            queue, {intent_key(first["intent"]): 120.0}, now=100.0)
        self.assertIs(selected, second)
        self.assertEqual(queue, [])

    def test_corridor_requires_three_distinct_updates(self):
        data = np.full((100, 120), 100, dtype=np.int16)
        # Ten-metre-long, two-metre-wide observed free corridor.
        data[40:60, 10:110] = 0
        grid = OccupancyMap(data, 0.10, 0.0, 0.0)
        parameters = GeometryParameters(
            corridor_width_min=1.0, corridor_width_max=4.0,
            corridor_forward_depth_min=4.0,
            corridor_confirmation_frames=3)
        detector = CorridorDetector(parameters)
        tracker = TemporalTopologyTracker(parameters)
        tracked = None
        for index in range(3):
            tracked = tracker.update_corridor(
                detector.detect(grid, (3.0 + 0.1 * index, 5.0)))
            self.assertIsNotNone(tracked)
            self.assertEqual(tracked.confirmed, index == 2)
        self.assertTrue(point_in_corridor((3.2, 5.0), tracked))
        self.assertFalse(corridor_entry_ready(
            (3.2, 5.0), tracked, bootstrap_origin=(3.0, 5.0),
            minimum_progress=2.0))
        self.assertTrue(corridor_entry_ready(
            (5.2, 5.0), tracked, bootstrap_origin=(3.0, 5.0),
            minimum_progress=2.0))
        self.assertTrue(bilateral_corridor_evidence(tracked))

    def test_one_sided_open_space_is_not_bilateral_corridor(self):
        class Corridor:
            confirmed = True
            estimated_width = 2.0
            forward_extent = 8.0
            confidence = 0.9
            centerline = [(0.0, 0.0), (8.0, 0.0)]
            left_wall = [(0.0, 1.0), (4.0, 1.0), (8.0, 1.0)]
            right_wall = [(0.0, -1.0), (0.5, -1.0), (1.0, -1.0)]
        self.assertFalse(bilateral_corridor_evidence(Corridor()))


if __name__ == "__main__":
    unittest.main()
