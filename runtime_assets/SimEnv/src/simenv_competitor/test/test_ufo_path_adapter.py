#!/usr/bin/env python3

import math
import os
import sys
import unittest

SCRIPT_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "scripts", "ufoexplorer_interface"))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from ufo_path_core import (crop_horizon, resample_path,
                           select_execution_waypoints, trim_path_to_pose)


class UfoPathCoreTest(unittest.TestCase):
    def test_trim_projects_and_removes_prefix(self):
        result = trim_path_to_pose([(0, 0), (1, 0), (2, 0), (3, 0)],
                                   (1.4, 0.2))
        self.assertAlmostEqual(result[0][0], 1.4)
        self.assertNotIn((0.0, 0.0), result)
        self.assertEqual(result[-1], (3.0, 0.0))

    def test_resample_and_horizon(self):
        sampled = resample_path([(0, 0), (4, 0)], 0.45)
        cropped = crop_horizon(sampled, 3.0)
        self.assertLessEqual(cropped[-1][0], 3.0 + 1e-6)
        self.assertGreaterEqual(len(cropped), 6)

    def test_skip_only_near_prefix_and_keep_sequence(self):
        sampled = [(0, 0), (.45, 0), (.9, 0), (1.35, 0)]
        selected = select_execution_waypoints(sampled, (0, 0), .6)
        self.assertEqual([round(point[0], 2) for point in selected],
                         [.9, 1.35])
        self.assertTrue(all(abs(point[2]) < 1e-9 for point in selected))

    def test_tangent_yaw(self):
        selected = select_execution_waypoints(
            [(0, 0), (0, .7), (0, 1.2)], (0, 0), .6)
        self.assertAlmostEqual(selected[0][2], math.pi / 2.0)


if __name__ == "__main__":
    unittest.main()
