#!/usr/bin/env python3

import os
import sys
import unittest

import numpy as np

SCRIPT_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "scripts"))
UFO_DIR = os.path.join(SCRIPT_DIR, "ufoexplorer_interface")
for path in (SCRIPT_DIR, UFO_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from baseline_planning_core import OccupancyGrid2D
from corridor_semantic_core import (corridor_observations_match,
                                    detect_parallel_wall_corridor,
                                    score_path_for_corridor)


class CorridorSemanticTest(unittest.TestCase):
    def setUp(self):
        data = np.full((200, 200), -1, dtype=np.int16)
        grid = OccupancyGrid2D(data, 0.1, -10.0, -10.0)
        # Online-looking vertical corridor: two balanced, long occupied walls
        # with a known-free strip and no layout/room metadata.
        for y_world in np.arange(-4.0, 5.01, 0.1):
            for x_world in np.arange(-0.9, 0.91, 0.1):
                x, y = grid.world_to_cell((x_world, y_world))
                grid.data[y, x] = 0
        for y_world in np.arange(-4.0, 5.01, 0.1):
            for x_world in (-1.0, 1.0):
                x, y = grid.world_to_cell((x_world, y_world))
                grid.data[y, x] = 100
        self.grid = grid

    def test_detects_parallel_wall_strip(self):
        result = detect_parallel_wall_corridor(self.grid, (0.0, -5.0))
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result["visible_length"], 4.0)
        self.assertAlmostEqual(result["width"], 2.0, delta=0.35)
        self.assertGreater(abs(result["axis"][1]), 0.9)

    def test_requires_consistent_observations(self):
        first = detect_parallel_wall_corridor(self.grid, (0.0, -5.0))
        second = detect_parallel_wall_corridor(self.grid, (0.05, -4.95))
        self.assertTrue(corridor_observations_match(first, second))

    def test_existing_path_toward_corridor_scores_higher(self):
        corridor = detect_parallel_wall_corridor(self.grid, (0.0, -5.0))
        toward = [(0.0, -4.5), (0.0, -4.0), (0.0, -3.5)]
        away = [(0.0, -5.5), (0.0, -6.0), (0.0, -6.5)]
        self.assertGreater(score_path_for_corridor(
            toward, (0.0, -5.0), corridor),
            score_path_for_corridor(away, (0.0, -5.0), corridor))


if __name__ == "__main__":
    unittest.main()
