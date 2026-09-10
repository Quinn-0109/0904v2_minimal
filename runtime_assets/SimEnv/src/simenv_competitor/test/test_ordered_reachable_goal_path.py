#!/usr/bin/env python3
"""Contract tests for ordered reachable-goal selection."""

import os
import sys
import unittest

import numpy as np


SCRIPT_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "scripts"))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from baseline_planning_core import (  # noqa: E402
    OccupancyGrid2D, astar_safe_path, ordered_reachable_goal_path)


def cell_point(x, y, offset_x=0.5, offset_y=0.5):
    """Return a world point inside a unit-resolution grid cell."""
    return float(x) + offset_x, float(y) + offset_y


def make_grid(data):
    return OccupancyGrid2D(
        np.asarray(data, dtype=np.int16), 1.0, 0.0, 0.0)


class OrderedReachableGoalPathTest(unittest.TestCase):
    def assert_matches_sequential(
            self, grid, start, goals, clearance_radius=0.0,
            reached_tolerance=0.05, allow_blocked_start=False):
        """Assert the selected result is exactly the first successful A*."""
        sequential = []
        selected_index = None
        selected_result = None
        for index, goal in enumerate(goals):
            candidate = astar_safe_path(
                grid, start, goal, clearance_radius, reached_tolerance,
                allow_blocked_start=allow_blocked_start)
            sequential.append(candidate)
            if candidate["success"]:
                selected_index = index
                selected_result = candidate
                break

        result = ordered_reachable_goal_path(
            grid, start, goals, clearance_radius, reached_tolerance,
            allow_blocked_start=allow_blocked_start)
        if selected_result is None:
            expected_success = False
            expected_reason = "no_reachable_goal"
            expected_path = []
            expected_astar_expansions = 0
        else:
            expected_success = selected_result["success"]
            expected_reason = selected_result["reason"]
            expected_path = selected_result["path"]
            expected_astar_expansions = selected_result.get("expansions", 0)

        self.assertEqual(result["goal_index"], selected_index)
        self.assertEqual(result["success"], expected_success)
        self.assertEqual(result["reason"], expected_reason)
        self.assertEqual(result["path"], expected_path)
        self.assertEqual(
            result["astar_expansions"], expected_astar_expansions)
        if selected_index is None:
            self.assertIsNone(result["goal"])
        else:
            self.assertEqual(result["goal"], goals[selected_index])
        return result, sequential

    def test_first_candidate_reachable(self):
        grid = make_grid(np.zeros((6, 8), dtype=np.int16))
        start = cell_point(1, 2)
        goals = [cell_point(6, 2), cell_point(6, 4)]

        result, _ = self.assert_matches_sequential(grid, start, goals)

        self.assertEqual(result["goal_index"], 0)
        self.assertEqual(result["candidate_reasons"], ["path_found", None])
        self.assertGreater(result["component_expansions"], 0)

    def test_blocked_and_unreachable_candidates_precede_reachable(self):
        data = np.zeros((7, 8), dtype=np.int16)
        data[:, 4] = 100
        grid = make_grid(data)
        start = cell_point(1, 1)
        goals = [
            cell_point(4, 1),  # Occupied endpoint.
            cell_point(6, 1),  # Free endpoint beyond an impassable wall.
            cell_point(2, 5),  # Reachable in the start component.
        ]

        result, sequential = self.assert_matches_sequential(
            grid, start, goals)

        self.assertEqual(
            [item["reason"] for item in sequential],
            ["goal_footprint_blocked", "goal_unreachable", "path_found"])
        self.assertEqual(result["goal_index"], 2)
        self.assertEqual(result["candidate_reasons"], [
            "goal_footprint_blocked", "goal_unreachable", "path_found"])

    def test_unknown_band_is_not_traversable(self):
        data = np.zeros((6, 8), dtype=np.int16)
        data[:, 3] = -1
        grid = make_grid(data)
        start = cell_point(1, 2)
        goals = [cell_point(6, 2), cell_point(1, 4)]

        result, _ = self.assert_matches_sequential(grid, start, goals)

        self.assertEqual(result["goal_index"], 1)
        self.assertEqual(
            result["candidate_reasons"],
            ["goal_unreachable", "path_found"])

    def test_diagonal_contact_does_not_connect_components(self):
        data = np.full((4, 4), 100, dtype=np.int16)
        data[1, 1] = 0
        data[2, 2] = 0
        grid = make_grid(data)
        start = cell_point(1, 1)
        goals = [cell_point(2, 2)]

        result, sequential = self.assert_matches_sequential(
            grid, start, goals)

        self.assertEqual(sequential[0]["reason"], "goal_unreachable")
        self.assertIsNone(result["goal_index"])
        self.assertEqual(result["candidate_reasons"], ["goal_unreachable"])
        self.assertEqual(result["astar_expansions"], 0)

    def test_allow_blocked_start_matches_astar_override(self):
        data = np.zeros((5, 7), dtype=np.int16)
        data[2, 1] = 100
        grid = make_grid(data)
        start = cell_point(1, 2)
        goals = [cell_point(5, 2)]

        disabled, _ = self.assert_matches_sequential(
            grid, start, goals, allow_blocked_start=False)
        enabled, _ = self.assert_matches_sequential(
            grid, start, goals, allow_blocked_start=True)

        self.assertFalse(disabled["success"])
        self.assertEqual(
            disabled["candidate_reasons"], ["start_footprint_blocked"])
        self.assertTrue(enabled["success"])
        self.assertEqual(enabled["reason"], "path_found_start_override")
        self.assertGreater(enabled["astar_expansions"], 0)

    def test_near_goal_keeps_astar_short_circuit(self):
        data = np.zeros((4, 4), dtype=np.int16)
        data[1, 1] = 100
        grid = make_grid(data)
        start = cell_point(1, 1)
        goal = (start[0] + 0.10, start[1])

        result, _ = self.assert_matches_sequential(
            grid, start, [goal], reached_tolerance=0.20)

        self.assertEqual(result["goal_index"], 0)
        self.assertEqual(result["reason"], "already_at_goal")
        self.assertEqual(result["path"], [goal])
        self.assertEqual(result["component_expansions"], 0)
        self.assertEqual(result["astar_expansions"], 0)

    def test_duplicate_goal_cell_preserves_priority_and_exact_endpoint(self):
        grid = make_grid(np.zeros((5, 7), dtype=np.int16))
        start = cell_point(1, 1)
        goals = [
            cell_point(5, 2, 0.20, 0.20),
            cell_point(5, 2, 0.80, 0.80),
        ]

        result, _ = self.assert_matches_sequential(grid, start, goals)

        self.assertEqual(result["goal_index"], 0)
        self.assertEqual(result["goal"], goals[0])
        self.assertEqual(result["path"][-1], goals[0])
        self.assertEqual(result["candidate_reasons"], ["path_found", None])

    def test_fixed_seed_random_maps_match_sequential_astar(self):
        rng = np.random.RandomState(20260820)
        for case_index in range(36):
            height = int(rng.randint(5, 10))
            width = int(rng.randint(5, 11))
            data = rng.choice(
                np.array([-1, 0, 100], dtype=np.int16),
                size=(height, width), p=(0.18, 0.64, 0.18))
            grid = make_grid(data)
            start_cell = (int(rng.randint(width)), int(rng.randint(height)))
            start = cell_point(
                start_cell[0], start_cell[1],
                float(rng.uniform(0.15, 0.85)),
                float(rng.uniform(0.15, 0.85)))
            goals = []
            for _ in range(int(rng.randint(1, 6))):
                kind = float(rng.uniform())
                if kind < 0.12:
                    goals.append((-0.25, float(rng.uniform(0.0, height))))
                elif kind < 0.24:
                    goals.append((
                        start[0] + float(rng.uniform(-0.20, 0.20)),
                        start[1] + float(rng.uniform(-0.20, 0.20))))
                elif kind < 0.36 and goals:
                    cell = grid.world_to_cell(goals[-1])
                    if cell is None:
                        goals.append(goals[-1])
                    else:
                        goals.append(cell_point(
                            cell[0], cell[1],
                            float(rng.uniform(0.10, 0.90)),
                            float(rng.uniform(0.10, 0.90))))
                else:
                    goals.append(cell_point(
                        int(rng.randint(width)), int(rng.randint(height)),
                        float(rng.uniform(0.10, 0.90)),
                        float(rng.uniform(0.10, 0.90))))
            clearance = float(rng.choice([0.0, 0.49, 1.0]))
            tolerance = float(rng.choice([0.05, 0.25, 0.75]))
            allow_blocked_start = bool(rng.randint(2))

            with self.subTest(case=case_index):
                self.assert_matches_sequential(
                    grid, start, goals,
                    clearance_radius=clearance,
                    reached_tolerance=tolerance,
                    allow_blocked_start=allow_blocked_start)


if __name__ == "__main__":
    unittest.main()
