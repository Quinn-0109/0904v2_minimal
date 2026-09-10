#!/usr/bin/env python3

import math
import os
import sys
import unittest

SCRIPTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, SCRIPTS)

from baseline_planning_core import (  # noqa: E402
    corridor_line_lateral_error, rebase_corridor_line_laterally,
)


class CorridorCenterlineRebaseTest(unittest.TestCase):
    def test_run45_like_lateral_offset_is_removed(self):
        origin = (7.90173089636649, -0.05814066253416211)
        axis = (0.9801577584429594, -0.19821899143137964)
        observed_exit_center = (14.288762748778954, -0.09539208125815815)

        self.assertGreater(corridor_line_lateral_error(
            observed_exit_center, origin, axis), 1.2)
        rebased = rebase_corridor_line_laterally(
            origin, axis, observed_exit_center)
        self.assertLess(corridor_line_lateral_error(
            observed_exit_center, rebased, axis), 1e-9)

    def test_rebase_preserves_all_longitudinal_station_coordinates(self):
        origin = (1.0, 2.0)
        axis = (math.cos(0.3), math.sin(0.3))
        observed_center = (4.0, 5.0)
        rebased = rebase_corridor_line_laterally(
            origin, axis, observed_center)

        for point in ((0.0, 0.0), (4.0, 5.0), (20.0, -3.0)):
            old_station = ((point[0] - origin[0]) * axis[0] +
                           (point[1] - origin[1]) * axis[1])
            new_station = ((point[0] - rebased[0]) * axis[0] +
                           (point[1] - rebased[1]) * axis[1])
            self.assertAlmostEqual(old_station, new_station, places=9)

    def test_invalid_axis_is_rejected(self):
        with self.assertRaises(ValueError):
            rebase_corridor_line_laterally((0.0, 0.0), (0.0, 0.0), (1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
