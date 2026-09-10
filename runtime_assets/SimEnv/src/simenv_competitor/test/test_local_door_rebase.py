#!/usr/bin/env python3

import os
import sys
import threading
import unittest
from collections import defaultdict
from types import SimpleNamespace


SCRIPT_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "scripts"))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from local_doorway_detector import LocalDoorwayDetector


def odometry(x, y, z=5.5):
    return SimpleNamespace(pose=SimpleNamespace(pose=SimpleNamespace(
        position=SimpleNamespace(x=x, y=y, z=z),
        orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0))))


class LocalDoorRebaseTest(unittest.TestCase):
    def detector(self):
        detector = LocalDoorwayDetector.__new__(LocalDoorwayDetector)
        detector.lock = threading.RLock()
        detector.pose = (20.0, -30.0, -40.0, 0.0)
        detector.pose_rebase_reset_distance = 4.0
        detector.trajectory = [(1.0, 19.0, -30.0)]
        detector.tracks = {(1, 1): {"seen": 1.0}}
        detector.observations = defaultdict(int, {(1, 1): 4})
        detector.elapsed = lambda: 2.0
        return detector

    def test_multi_metre_truth_rebase_starts_new_geometry_epoch(self):
        detector = self.detector()
        detector._on_odom(odometry(0.0, 13.3))
        self.assertEqual(detector.tracks, {})
        self.assertEqual(dict(detector.observations), {})
        self.assertEqual(detector.trajectory, [(2.0, 0.0, 13.3)])

    def test_normal_motion_retains_tracks(self):
        detector = self.detector()
        detector._on_odom(odometry(21.0, -30.0))
        self.assertIn((1, 1), detector.tracks)
        self.assertEqual(detector.observations[(1, 1)], 4)


if __name__ == "__main__":
    unittest.main()
