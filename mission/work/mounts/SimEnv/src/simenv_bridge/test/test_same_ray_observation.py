"""Observe same-ray track pairs; decide nothing.

Seed 20's floor_1_room_2 G4 scan is why.  Two pairs sit inside the same
bearing/ratio window, measured from the viewpoint at (-3.8896, 29.5173):

  far track 4 (0.406 m from sphere_00)  <- near D  0.571 deg  1.619x
  far track 5 (1.560 m from sphere_01)  <- near C  0.212 deg  1.184x

Acting on the second recovers a miss; acting on the first destroys a good
detection.  Bearing and ratio cannot tell them apart, so this records what
might: which frames fed each member, and whether the near member recurs.
"""
import math
from types import SimpleNamespace
import unittest

from test_danger_detector import _load_module


VIEWPOINT_G4 = [-3.8896, 29.5173]
TRACK_4 = (-6.9865, 23.1310)     # 5 frames, 0.406 m from danger_red_sphere_00
TRACK_5 = (-5.3855, 21.9878)     # 4 frames, 1.560 m from danger_red_sphere_01
NEAR_C = (-5.1299, 23.1511)      # 0.370 m from danger_red_sphere_01
NEAR_D = (-5.7629, 25.5542)      # 2.152 m from the nearest sphere


def track(track_id, position, frame_ids, role="G4",
          waypoint="floor_1_room_2_g4", observer=None):
    return {
        "id": track_id,
        "x": position[0],
        "y": position[1],
        "z": 2.75,
        "dimensions": 3,
        "count": len(frame_ids),
        "frame_ids": list(frame_ids),
        "metadata": {
            "floor": 2,
            "room_id": "floor_1_room_2",
            "viewpoint_role": role,
            "waypoint": waypoint,
            "observer_xy": list(VIEWPOINT_G4 if observer is None else observer),
        },
    }


class SameRayObservationTest(unittest.TestCase):
    def setUp(self):
        self.module = _load_module()
        self.associate = self.module.same_ray_track_associations

    def _seed20(self, c_frames, d_frames):
        return [
            track(4, TRACK_4, [10, 12, 14, 16, 18]),
            track(5, TRACK_5, [11, 13, 15, 17]),
            track(6, NEAR_C, c_frames),
            track(7, NEAR_D, d_frames),
        ]

    def test_both_measured_pairs_are_reported(self):
        found = self.associate(self._seed20([19], [20]), confirm_count=3)
        pairs = {(item["far_track_id"], item["near_track_id"]): item
                 for item in found}

        self.assertIn((4, 7), pairs)
        self.assertIn((5, 6), pairs)
        self.assertAlmostEqual(pairs[(4, 7)]["bearing_gap_deg"], 0.571, 2)
        self.assertAlmostEqual(pairs[(4, 7)]["range_ratio"], 1.619, 2)
        self.assertAlmostEqual(pairs[(5, 6)]["bearing_gap_deg"], 0.212, 2)
        self.assertAlmostEqual(pairs[(5, 6)]["range_ratio"], 1.184, 2)

    def test_a_single_frame_near_member_does_not_recur(self):
        found = self.associate(self._seed20([19], [20]), confirm_count=3)
        for item in found:
            self.assertFalse(item["near_recurs_independently"])
            self.assertFalse(item["near_confirmed"])

    def test_a_recurring_near_member_is_flagged(self):
        found = self.associate(self._seed20([19, 21], [20]), confirm_count=3)
        pairs = {item["near_track_id"]: item for item in found}

        self.assertTrue(pairs[6]["near_recurs_independently"])
        self.assertFalse(pairs[7]["near_recurs_independently"])

    def test_shared_frames_are_reported_separately_from_disjoint_ones(self):
        # Same frames feeding both members means two things seen at once.
        found = self.associate(self._seed20([11, 13], [20]), confirm_count=3)
        pairs = {item["near_track_id"]: item for item in found}

        self.assertEqual(pairs[6]["shared_frame_ids"], [11, 13])
        self.assertFalse(pairs[6]["frames_are_disjoint"])
        self.assertEqual(pairs[7]["shared_frame_ids"], [])
        self.assertTrue(pairs[7]["frames_are_disjoint"])

    def test_both_frame_lists_are_carried_so_the_pair_can_be_audited(self):
        found = self.associate(self._seed20([19, 21], [20]), confirm_count=3)
        item = [pair for pair in found if pair["near_track_id"] == 6][0]

        self.assertEqual(item["far_frame_ids"], [11, 13, 15, 17])
        self.assertEqual(item["near_frame_ids"], [19, 21])
        self.assertEqual(item["far_frames"], 4)
        self.assertEqual(item["near_frames"], 2)
        self.assertTrue(item["far_confirmed"])

    def test_the_nearer_member_is_always_the_near_side(self):
        for tracks in (self._seed20([19], [20]),
                       list(reversed(self._seed20([19], [20])))):
            for item in self.associate(tracks, confirm_count=3):
                self.assertLessEqual(item["near_range_m"], item["far_range_m"])
                self.assertGreaterEqual(item["range_ratio"], 1.10)

    def test_a_pair_across_viewpoints_is_not_an_association(self):
        tracks = [track(5, TRACK_5, [11, 13, 15, 17]),
                  track(6, NEAR_C, [19], role="G3",
                        waypoint="floor_1_room_2_g3",
                        observer=[-5.7666, 27.5628])]
        self.assertEqual(self.associate(tracks, confirm_count=3), [])

    def test_a_track_without_an_observer_is_not_reported(self):
        stale = track(6, NEAR_C, [19])
        stale["metadata"].pop("observer_xy")
        tracks = [track(5, TRACK_5, [11, 13, 15, 17]), stale]
        self.assertEqual(self.associate(tracks, confirm_count=3), [])

    def test_tracks_off_each_others_bearing_are_not_reported(self):
        tracks = [track(4, TRACK_4, [10, 12, 14]),
                  track(5, TRACK_5, [11, 13, 15])]
        self.assertEqual(self.associate(tracks, confirm_count=3), [])


class TrackFrameIdentityTest(unittest.TestCase):
    """count is already a per-frame count; the batch point totals are not."""

    def setUp(self):
        self.module = _load_module()

    def test_a_track_takes_at_most_one_candidate_per_frame(self):
        tracker = self.module.DetectionTracker(merge_radius=0.75,
                                               confirm_count=3)
        # Two candidates 0.10 m apart in one frame: the second cannot be
        # merged into the track the first opened, so it opens its own.
        tracker.update([(0.0, 0.0, 2.75), (0.10, 0.0, 2.75)],
                       metadata={}, frame_id=7)
        self.assertEqual(len(tracker.tracks), 2)
        for item in tracker.tracks:
            self.assertEqual(item["count"], 1)
            self.assertEqual(item["frame_ids"], [7])

    def test_frame_ids_record_which_frames_fed_a_track(self):
        tracker = self.module.DetectionTracker(merge_radius=0.75,
                                               confirm_count=3)
        for frame_id in (4, 9, 11):
            tracker.update([(1.0, 1.0, 2.75)], metadata={}, frame_id=frame_id)
        self.assertEqual(tracker.tracks[0]["frame_ids"], [4, 9, 11])
        self.assertEqual(tracker.tracks[0]["count"], 3)

    def test_assignments_name_the_track_each_point_fed(self):
        tracker = self.module.DetectionTracker(merge_radius=0.75,
                                               confirm_count=3)
        tracker.update([(0.0, 0.0, 2.75), (5.0, 5.0, 2.75)],
                       metadata={}, frame_id=1)
        self.assertEqual([(index, track_id) for index, track_id, _c, _n
                          in tracker.last_assignments], [(0, 1), (1, 2)])

    def test_tracking_is_unchanged_when_no_frame_id_is_given(self):
        tracker = self.module.DetectionTracker(merge_radius=0.75,
                                               confirm_count=3)
        for _ in range(3):
            tracker.update([(2.0, 2.0, 2.75)])
        self.assertEqual(tracker.tracks[0]["count"], 3)
        self.assertEqual(tracker.tracks[0]["frame_ids"], [])
        self.assertEqual(len(tracker.confirmed()), 1)


class ProjectionProvenanceTest(unittest.TestCase):
    """world_points is filtered twice after the candidates are built.

    The door half-plane and the room-boundary filters both drop points, so
    an assignment index is an index into the FILTERED list.  Matching a
    record to its assignment by position rather than by build order is what
    keeps the provenance attached to the right track.
    """

    def setUp(self):
        self.module = _load_module()
        self.detector = self.module.Detector.__new__(self.module.Detector)
        self.detector.frames_processed = 42
        self.detector.base_xyz = [-3.8896, 29.5173, 2.9152]
        self.detector._projection_provenance_limit = 400
        self.detector._scan_batch = {}

    IMAGE_TIME_POSE = [-3.8901, 29.5168]

    def _record(self, world_points, assignments, records):
        self.detector.tracker = SimpleNamespace(
            last_assignments=assignments)
        self.detector._record_projection_provenance(
            world_points, records, self.IMAGE_TIME_POSE)
        return self.detector._scan_batch.get("projection_provenance", [])

    def test_a_filtered_out_candidate_does_not_shift_the_rest(self):
        # Three candidates were built; the middle one was dropped by a
        # filter, so the surviving points are at indices 0 and 1.
        records = [
            {"world_xy": [-5.1299, 23.1511], "projection": "depth_only"},
            {"world_xy": [99.0, 99.0], "projection": "ground_only"},
            {"world_xy": [-5.3855, 21.9878], "projection": "fused_midpoint"},
        ]
        world_points = [(-5.1299, 23.1511, 2.75), (-5.3855, 21.9878, 2.75)]
        stored = self._record(world_points,
                              [(0, 6, 1, True), (1, 5, 4, False)], records)

        self.assertEqual([item["track_id"] for item in stored], [6, 5])
        self.assertEqual([item["projection"] for item in stored],
                         ["depth_only", "fused_midpoint"])
        self.assertEqual([item["world_xy"] for item in stored],
                         [[-5.1299, 23.1511], [-5.3855, 21.9878]])

    def test_each_entry_carries_the_frame_and_the_observer(self):
        records = [{"world_xy": [-5.3855, 21.9878],
                    "projection": "fused_midpoint"}]
        stored = self._record([(-5.3855, 21.9878, 2.75)],
                              [(0, 5, 4, False)], records)

        self.assertEqual(stored[0]["frame_id"], 42)
        self.assertEqual(stored[0]["track_count_after"], 4)
        self.assertFalse(stored[0]["opened_track"])
        # The image-time pose the points were projected from, which is
        # what the track metadata carries -- not self.base_xyz.
        self.assertEqual(stored[0]["observer_xy"], self.IMAGE_TIME_POSE)
        self.assertNotEqual(stored[0]["observer_xy"],
                            self.detector.base_xyz[:2])

    def test_an_out_of_range_assignment_is_skipped(self):
        records = [{"world_xy": [1.0, 2.0], "projection": "depth_only"}]
        stored = self._record([(1.0, 2.0, 2.75)],
                              [(0, 5, 1, True), (7, 9, 1, True)], records)
        self.assertEqual(len(stored), 1)

    def test_the_record_is_capped_and_says_so(self):
        self.detector._projection_provenance_limit = 2
        records = [{"world_xy": [float(i), 0.0], "projection": "depth_only"}
                   for i in range(5)]
        points = [(float(i), 0.0, 2.75) for i in range(5)]
        stored = self._record(
            points, [(i, i + 1, 1, True) for i in range(5)], records)

        self.assertEqual(len(stored), 2)
        self.assertTrue(
            self.detector._scan_batch["projection_provenance_truncated"])

    def test_nothing_is_recorded_without_a_scan_batch(self):
        self.detector._scan_batch = None
        records = [{"world_xy": [1.0, 2.0], "projection": "depth_only"}]
        self.detector.tracker = SimpleNamespace(
            last_assignments=[(0, 5, 1, True)])
        self.detector._record_projection_provenance(
            [(1.0, 2.0, 2.75)], records,
            self.IMAGE_TIME_POSE)   # must not raise


if __name__ == "__main__":
    unittest.main()
