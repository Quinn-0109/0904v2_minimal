"""Seed 1001's second run still reported a false positive in floor_1_room_3.

The G3 scan confirmed two tracks 37 ms apart: the sphere at 5.49 m from the
viewpoint and a second "source" at 7.79 m.  Their bearings from that
viewpoint differ by 0.15 deg, and the range ratio, 1.42, is the camera
height over the sphere centre -- the far track is the camera ray's own
intersection with the floor behind the ball.  No radius gate can reach it:
the same room's two genuine spheres stood 2.08 m apart and this pair stands
2.30 m apart.

Measured in fwd1001_s1001_20260912_151057.
"""
import math
import unittest

from test_danger_detector import _load_module


VIEWPOINT_G3 = [6.0319, 29.5732]
SPHERE_TRACK = [2.9391, 25.0378, 2.75]
GROUND_GHOST = [1.6243, 23.1459, 2.75]
SPHERE_01_TRACK = [9.0596, 32.6731, 2.75]
SPHERE_03 = (2.940, 25.129)


def event(track_id, position, frames, role="G3", waypoint="floor_1_room_3_g3",
          observer=None):
    return {
        "track_id": track_id,
        "position": list(position),
        "evidence_frames": frames,
        "floor": 2,
        "room_id": "floor_1_room_3",
        "viewpoint_role": role,
        "waypoint": waypoint,
        "observer_xy": list(VIEWPOINT_G3 if observer is None else observer),
    }


class SameRayDuplicateTest(unittest.TestCase):
    def setUp(self):
        self.module = _load_module()
        self.losers = self.module.same_ray_duplicate_losers

    def test_measured_pair_shares_one_bearing(self):
        def bearing(point):
            return math.degrees(math.atan2(point[1] - VIEWPOINT_G3[1],
                                           point[0] - VIEWPOINT_G3[0]))

        self.assertLess(
            abs(bearing(SPHERE_TRACK) - bearing(GROUND_GHOST)), 0.2)

    def test_the_farther_track_is_dropped(self):
        detections = [event(3, SPHERE_TRACK, 9), event(4, GROUND_GHOST, 3)]
        self.assertEqual(self.losers(detections), {1})

    def test_order_does_not_decide_the_survivor(self):
        detections = [event(4, GROUND_GHOST, 3), event(3, SPHERE_TRACK, 9)]
        self.assertEqual(self.losers(detections), {0})

    def test_the_ghost_is_dropped_even_with_more_frames(self):
        detections = [event(3, SPHERE_TRACK, 3), event(4, GROUND_GHOST, 9)]
        self.assertEqual(self.losers(detections), {1})

    def test_a_second_sphere_off_the_ray_is_kept(self):
        detections = [event(3, SPHERE_TRACK, 9), event(6, SPHERE_01_TRACK, 7)]
        self.assertEqual(self.losers(detections), set())

    def test_two_spheres_two_metres_apart_are_kept(self):
        # The pair the radius gates were tuned against: 2.08 m apart and
        # confirmed together.  Placed across the viewing direction they sit
        # 20 deg apart, far outside the ray window.
        other = [SPHERE_TRACK[0] - 1.7, SPHERE_TRACK[1] + 1.2, 2.75]
        detections = [event(3, SPHERE_TRACK, 9), event(5, other, 4)]
        self.assertEqual(self.losers(detections), set())

    def test_a_different_viewpoint_is_never_merged(self):
        detections = [
            event(3, SPHERE_TRACK, 9),
            event(4, GROUND_GHOST, 3, role="G4",
                  waypoint="floor_1_room_3_g4"),
        ]
        self.assertEqual(self.losers(detections), set())

    def test_a_track_without_an_observer_is_kept(self):
        stale = event(4, GROUND_GHOST, 3)
        stale.pop("observer_xy")
        self.assertEqual(self.losers([event(3, SPHERE_TRACK, 9), stale]),
                         set())

    def test_dropping_the_ghost_leaves_only_a_scoring_detection(self):
        detections = [event(3, SPHERE_TRACK, 9), event(4, GROUND_GHOST, 3)]
        kept = [item for index, item in enumerate(detections)
                if index not in self.losers(detections)]
        self.assertEqual(len(kept), 1)
        offset = math.hypot(kept[0]["position"][0] - SPHERE_03[0],
                            kept[0]["position"][1] - SPHERE_03[1])
        self.assertLess(offset, 1.5)


if __name__ == "__main__":
    unittest.main()
