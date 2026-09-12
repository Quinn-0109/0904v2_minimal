"""Seed 1001 scored one localisation error as a false positive and a miss.

floor_1_room_3 saw danger_red_sphere_03 from both viewpoints.  G3 built a
nine-frame track 1.73 m off the sphere; G4 built one 0.38 m from it.  The
cross-view merge decided they were the same ball, kept the G3 track because
it had more frames, and discarded G4's position with it.  The survivor fell
outside the 1.5 m matching tolerance, so the run reported a false positive
and never matched the sphere.

Measured in fwd1001_s1001_20260912_142014.
"""
import unittest

from test_danger_detector import _load_module


G3_TRACK = [2.1235, 23.6079, 2.75]
G4_TRACK = [2.8243, 24.7665, 2.75]
SPHERE_03 = (2.940, 25.129)
TOLERANCE_M = 1.5


def distance(position, truth):
    return ((position[0] - truth[0]) ** 2 +
            (position[1] - truth[1]) ** 2) ** 0.5


class CrossViewMergePositionTest(unittest.TestCase):
    def setUp(self):
        self.combine = _load_module().combine_cross_view_positions

    def test_the_view_that_was_thrown_away_is_the_one_that_matched(self):
        self.assertGreater(distance(G3_TRACK, SPHERE_03), TOLERANCE_M)
        self.assertLess(distance(G4_TRACK, SPHERE_03), TOLERANCE_M)

    def test_keeping_the_winner_reproduces_the_failure(self):
        position = self.combine(G3_TRACK, G4_TRACK, "winner")
        self.assertEqual(position[:2], G3_TRACK[:2])
        self.assertGreater(distance(position, SPHERE_03), TOLERANCE_M)

    def test_the_midpoint_brings_the_sphere_back_inside_tolerance(self):
        position = self.combine(G3_TRACK, G4_TRACK, "midpoint")
        self.assertLess(distance(position, SPHERE_03), TOLERANCE_M)
        self.assertLess(distance(position, SPHERE_03),
                        distance(G3_TRACK, SPHERE_03))

    def test_the_height_of_the_surviving_track_is_kept(self):
        position = self.combine(G3_TRACK, G4_TRACK, "midpoint")
        self.assertEqual(position[2], G3_TRACK[2])

    def test_a_position_without_a_partner_is_returned_unchanged(self):
        for partner in ([], [1.0], None):
            self.assertEqual(
                self.combine(G3_TRACK, partner, "midpoint"), G3_TRACK)


if __name__ == "__main__":
    unittest.main()
