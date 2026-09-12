"""The cross-view midpoint must not feed the same-view duplicate rules.

401ebac wrote the midpoint into the surviving record inside the cross-view
merge loop.  Everything downstream -- strong_same_view_duplicate_losers, the
weak D1 fallback and its room_depth tie-break -- then measured against a
coordinate no camera reported, so a weak duplicate the old code deleted can
survive a shift it had nothing to do with.  That is a false positive the
midpoint change was never meant to introduce.

The midpoint is a reporting decision.  It belongs after every duplicate
rule has run, on the record that survived them.
"""
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest

from test_danger_detector import _load_module


ROOM = "floor_1_room_9"
DOOR_PLANE_X = -5.0


def track(track_id, x, y, frames):
    return {"id": track_id, "x": x, "y": y, "z": 2.75, "count": frames}


def event(track_id, role, waypoint, observer, elapsed):
    return {
        "track_id": track_id,
        "floor": 2,
        "room_id": ROOM,
        "viewpoint_role": role,
        "waypoint": waypoint,
        "door_plane_x": DOOR_PLANE_X,
        "door_inward_direction": 1.0,
        "observer_xy": list(observer),
        "first_confirmed_elapsed_sec": elapsed,
    }


def run_write_output(tracks, events, all_tracks=None):
    module = _load_module()
    detector = module.Detector.__new__(module.Detector)
    detector._output_lock = threading.RLock()
    detector.terminal_status = None
    detector._exploration_elapsed = lambda: 300.0
    detector.started_wall_time = 0.0
    detector.frames_processed = 1000
    detector.frames_with_candidates = 10
    detector.scan_batches = []
    detector.detection_events = events
    detector.cross_view_merge_position = "midpoint"
    detector.tracker = SimpleNamespace(
        confirm_count=3,
        confirmed_tracks=lambda: tracks,
        tracks=list(tracks if all_tracks is None else all_tracks),
    )
    detector.rospy = SimpleNamespace(loginfo=lambda *_args: None)

    with tempfile.TemporaryDirectory() as directory:
        detector.out_json = str(Path(directory) / "detections.json")
        detector._write_output("completed")
        return json.loads(Path(detector.out_json).read_text("utf-8"))


class MergePositionOrderingTest(unittest.TestCase):
    def test_a_midpoint_does_not_rescue_a_weak_same_view_duplicate(self):
        # Strong G3 track and its three-frame same-view tail sit 1.35 m
        # apart, inside the 1.40 m same-view radius that deletes the tail.
        # The strong track also has a G4 partner 1.60 m away, so the merge
        # moves its reported position 0.80 m and the pair reads as 2.15 m.
        tracks = [track(1, 0.0, 0.0, 9),
                  track(2, 1.35, 0.0, 3),
                  track(3, -1.6, 0.0, 4)]
        events = [event(1, "G3", "wp_g3", (0.0, -6.0), 100.0),
                  event(2, "G3", "wp_g3", (0.0, -6.0), 100.0),
                  event(3, "G4", "wp_g4", (-1.6, -6.0), 101.0)]

        output = run_write_output(tracks, events)
        kept = sorted(item["track_id"] for item in output["detections"])

        self.assertEqual(kept, [1], "the weak same-view tail survived")
        self.assertEqual(output["confirmed_count"], 1)

    def test_the_survivor_still_reports_the_midpoint(self):
        tracks = [track(1, 0.0, 0.0, 9),
                  track(2, 1.35, 0.0, 3),
                  track(3, -1.6, 0.0, 4)]
        events = [event(1, "G3", "wp_g3", (0.0, -6.0), 100.0),
                  event(2, "G3", "wp_g3", (0.0, -6.0), 100.0),
                  event(3, "G4", "wp_g4", (-1.6, -6.0), 101.0)]

        survivor = run_write_output(tracks, events)["detections"][0]

        self.assertEqual(survivor["track_id"], 1)
        self.assertAlmostEqual(survivor["position"][0], -0.8, places=4)
        self.assertEqual(survivor["merged_source_positions"],
                         [[0.0, 0.0, 2.75], [-1.6, 0.0, 2.75]])
        self.assertAlmostEqual(
            survivor["merged_source_separation_m"], 1.6, places=4)


class Seed1001PipelineTest(unittest.TestCase):
    """The whole pipeline on the room that produced both scoring errors.

    floor_1_room_3 in fwd1001_s1001_20260912_151057: G3 saw the sphere and
    its own ground re-projection, G4 saw the sphere, and both scans also
    saw danger_red_sphere_01 across the room.
    """

    SPHERE_03 = (2.940, 25.129)

    def _room_3(self):
        def at(track_id, x, y, frames, role, waypoint, observer, elapsed):
            item = event(track_id, role, waypoint, observer, elapsed)
            item["door_plane_x"] = 1.1
            return track(track_id, x, y, frames), item

        rows = [
            at(3, 2.9391, 25.0378, 10, "G3", "floor_1_room_3_g3",
               (6.0319, 29.5732), 291.076),
            at(4, 1.6243, 23.1459, 3, "G3", "floor_1_room_3_g3",
               (6.0319, 29.5732), 291.113),
            at(5, 2.6428, 24.2426, 5, "G4", "floor_1_room_3_g4",
               (3.8825, 28.2168), 298.7),
            at(6, 8.7633, 32.3567, 7, "G4", "floor_1_room_3_g4",
               (3.8825, 28.2168), 298.758),
        ]
        return [row[0] for row in rows], [row[1] for row in rows]

    def test_the_room_reports_one_sphere_per_sphere(self):
        output = run_write_output(*self._room_3())
        kept = sorted(item["track_id"] for item in output["detections"])

        # 4 is the ground re-projection, 5 is G4's view of the same sphere
        # as 3.  Two spheres stood in this room and two records leave it.
        self.assertEqual(kept, [3, 6])

    def test_the_sphere_lands_inside_the_matching_tolerance(self):
        output = run_write_output(*self._room_3())
        survivor = [item for item in output["detections"]
                    if item["track_id"] == 3][0]
        offset = ((survivor["position"][0] - self.SPHERE_03[0]) ** 2 +
                  (survivor["position"][1] - self.SPHERE_03[1]) ** 2) ** 0.5

        self.assertIn("merged_source_positions", survivor)
        self.assertLess(offset, 1.5)
        self.assertAlmostEqual(offset, 0.511, places=2)


class ObservationIsAdditiveTest(unittest.TestCase):
    """The same-ray record must not reach any decision.

    Seed 20's shape: a four-frame track 1.560 m from its sphere with a
    one-frame track on its bearing 1.184x nearer.  The association is
    reported; the reported position is still the track's own.
    """

    VIEWPOINT_G4 = [-3.8896, 29.5173]

    def _room_2(self):
        def at(track_id, x, y, frames, frame_ids):
            item = event(track_id, "G4", "floor_1_room_2_g4",
                         self.VIEWPOINT_G4, 265.0 + track_id * 0.1)
            item["door_plane_x"] = -1.1
            item["door_inward_direction"] = -1.0
            row = track(track_id, x, y, frames)
            row["metadata"] = dict(item)
            row["frame_ids"] = list(frame_ids)
            return row, item

        rows = [at(5, -5.3855, 21.9878, 4, [11, 13, 15, 17]),
                at(6, -5.1299, 23.1511, 1, [19])]
        return [rows[0][0]], [rows[0][1]], [row[0] for row in rows]

    def test_the_position_reported_is_still_the_track_s_own(self):
        confirmed, events, every = self._room_2()
        output = run_write_output(confirmed, events, all_tracks=every)
        found = output["detections"]

        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["track_id"], 5)
        self.assertEqual(found[0]["position"][:2], [-5.3855, 21.9878])
        self.assertEqual(found[0]["evidence_frames"], 4)
        self.assertNotIn("same_ray_origin", found[0])

    def test_the_association_is_reported_alongside_it(self):
        confirmed, events, every = self._room_2()
        output = run_write_output(confirmed, events, all_tracks=every)

        self.assertEqual(output["same_ray_observation"], "record_only_v1")
        self.assertEqual(len(output["same_ray_associations"]), 1)
        pair = output["same_ray_associations"][0]
        self.assertEqual((pair["far_track_id"], pair["near_track_id"]), (5, 6))
        self.assertAlmostEqual(pair["range_ratio"], 1.184, places=2)
        self.assertEqual(pair["near_frame_ids"], [19])
        self.assertTrue(pair["frames_are_disjoint"])
        self.assertFalse(pair["near_recurs_independently"])

    def test_the_rules_revision_did_not_move(self):
        confirmed, events, every = self._room_2()
        output = run_write_output(confirmed, events, all_tracks=every)
        self.assertEqual(output["detector_rules_revision"],
                         "same_ray_before_merge_midpoint_after_dedup_v1")


if __name__ == "__main__":
    unittest.main()
