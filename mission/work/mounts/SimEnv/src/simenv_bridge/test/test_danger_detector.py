#!/usr/bin/env python3
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest

import cv2
import numpy as np


def _load_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "danger_detector.py"
    spec = importlib.util.spec_from_file_location("danger_detector_under_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DangerDetectorTest(unittest.TestCase):
    def test_partial_camera_evidence_requests_only_one_rescan_per_room(self):
        mod = _load_module()

        class Tracker:
            @staticmethod
            def confirmed_tracks():
                return []

        class Publisher:
            def __init__(self):
                self.messages = []

            def publish(self, message):
                self.messages.append(message.data)

        detector = mod.Detector.__new__(mod.Detector)
        detector.scan_active = False
        detector.scan_label = "floor_1_room_0_scan_g3"
        detector.stage = "floor_1_exploration"
        detector.floor = 1
        detector.room_id = "floor_1_room_0"
        detector.viewpoint_role = "G3"
        detector.scan_points = []
        detector.scan_frame = 0
        detector.scan_batches = []
        detector._scan_batch = None
        detector.scan_merge_radius = 1.2
        detector.tracker = Tracker()
        detector._danger_rescan_requested_rooms = set()
        detector._room_scan_evidence = {}
        detector.danger_rescan_pub = Publisher()
        detector.String = lambda data: SimpleNamespace(data=data)
        detector.rospy = SimpleNamespace(loginfo=lambda *_args: None)
        detector.terminal_status = None
        detector._write_output = lambda _status: None

        detector._scan_active_cb(SimpleNamespace(data=True))
        detector.scan_frame += 1
        detector.scan_points.append((2.0, 3.0, detector.scan_frame))
        detector._record_scan_frame([(2.0, 3.0, 0.15)])
        detector._scan_active_cb(SimpleNamespace(data=False))
        self.assertEqual(len(detector.danger_rescan_pub.messages), 0)

        # G4 sees nothing. The combined room evidence must still request the
        # single bounded rescan at the stronger G3 viewpoint.
        detector.scan_label = "floor_1_room_0_scan_g4"
        detector.viewpoint_role = "G4"
        detector._scan_active_cb(SimpleNamespace(data=True))
        detector._scan_active_cb(SimpleNamespace(data=False))

        self.assertEqual(len(detector.danger_rescan_pub.messages), 1)
        request = json.loads(detector.danger_rescan_pub.messages[0])
        self.assertEqual(request["room_id"], "floor_1_room_0")
        self.assertEqual(request["candidate_frames"], 0)
        self.assertEqual(request["combined_candidate_frames"], 1)
        self.assertEqual(request["preferred_viewpoint_role"], "G3")

    def test_terminal_callback_wins_concurrent_running_snapshot(self):
        mod = _load_module()

        class EmptyTracker:
            @staticmethod
            def confirmed_tracks():
                return []

            @staticmethod
            def confirmed():
                return []

        with tempfile.TemporaryDirectory() as directory:
            detector = mod.Detector.__new__(mod.Detector)
            detector.out_json = str(Path(directory) / "detections.json")
            detector._output_lock = threading.RLock()
            detector.start_time = None
            detector.started_wall_time = None
            detector.terminal_status = None
            detector.exploration_active = True
            detector.scan_active = False
            detector.detection_events = []
            detector.frames_processed = 42
            detector.frames_with_candidates = 3
            detector.tracker = EmptyTracker()
            detector.rospy = SimpleNamespace(loginfo=lambda *_args: None)
            detector.scan_batches = []

            barrier = threading.Barrier(3)
            errors = []

            def running_snapshot():
                try:
                    barrier.wait()
                    detector._exploration_active_cb(
                        SimpleNamespace(data=False))
                except Exception as error:  # pragma: no cover - assertion path
                    errors.append(error)

            def terminal_snapshot():
                try:
                    barrier.wait()
                    detector._finalize("completed")
                except Exception as error:  # pragma: no cover - assertion path
                    errors.append(error)

            threads = [
                threading.Thread(target=running_snapshot),
                threading.Thread(target=terminal_snapshot),
            ]
            for thread in threads:
                thread.start()
            barrier.wait()
            for thread in threads:
                thread.join()

            self.assertEqual(errors, [])
            payload = json.loads(Path(detector.out_json).read_text())
            self.assertEqual(payload["status"], "completed")
            self.assertEqual(list(Path(directory).glob("*.tmp.*")), [])

    def test_scan_batch_counters_snapshot_and_serialize_without_ros(self):
        mod = _load_module()

        class RecordingTracker:
            def __init__(self):
                self.update_calls = []

            def update(self, points, metadata=None):
                self.update_calls.append(list(points))
                return []

            def confirmed_tracks(self):
                return []

            def confirmed(self):
                return []

        with tempfile.TemporaryDirectory() as directory:
            detector = mod.Detector.__new__(mod.Detector)
            detector.out_json = str(Path(directory) / "detections.json")
            detector._output_lock = threading.RLock()
            detector.start_time = None
            detector.started_wall_time = None
            detector.terminal_status = None
            detector.exploration_active = True
            detector.scan_active = False
            detector.detection_events = []
            detector.frames_processed = 0
            detector.frames_with_candidates = 0
            detector.tracker = RecordingTracker()
            detector.scan_points = []
            detector.scan_frame = 0
            detector.scan_batches = []
            detector._scan_batch = None
            detector.scan_merge_radius = 1.2
            # Frozen threshold must come from the start-of-scan label
            # (room_LF_scan -> 1), not the mid-scan relabel below (-> 3).
            detector.scan_label = "room_LF_scan"
            detector.stage = "floor_1_exploration"
            detector.floor = 1
            log_messages = []
            detector.rospy = SimpleNamespace(
                loginfo=lambda *args: log_messages.append(
                    str(args[0]) % tuple(args[1:]) if len(args) > 1 else str(args[0])))

            detector._scan_active_cb(SimpleNamespace(data=True))
            # Later route-state callbacks must not relabel the open batch.
            detector.scan_label = "floor_1_room_2_scan_b"
            detector.stage = "floor_2_exploration"
            detector.floor = 2
            for points in (
                [(3.0, 4.0, 2.15)],
                [(3.05, 4.0, 2.15)],
                [(5.0, 6.0, 2.15)],
                [],
                [(2.95, 4.0, 2.15)],
            ):
                if points:
                    detector.scan_frame += 1
                    detector.scan_points.extend(
                        (point[0], point[1], detector.scan_frame)
                        for point in points)
                detector._record_scan_frame(points)
            detector._scan_active_cb(SimpleNamespace(data=False))

            self.assertEqual(detector.tracker.update_calls, [])
            payload = json.loads(Path(detector.out_json).read_text())
            self.assertEqual(len(payload["scan_batches"]), 1)
            batch = payload["scan_batches"][0]
            self.assertEqual(batch["label"], "room_LF_scan")
            self.assertEqual(batch["stage"], "floor_1_exploration")
            self.assertEqual(batch["floor"], 1)
            self.assertEqual(batch["processed_frames"], 5)
            self.assertEqual(batch["candidate_frames"], 4)
            self.assertEqual(batch["raw_candidate_points"], 4)
            self.assertEqual(batch["minimum_cluster_points"], 1)
            # With the frozen room_LF_scan threshold of 1, both the merged
            # (3.0/3.05/2.95, 4.0) cluster and the lone (5.0, 6.0) candidate
            # survive; the mid-scan relabel to a 3-point threshold must not
            # change that.
            self.assertEqual(batch["clustered_candidate_count"], 2)
            self.assertEqual(batch["candidate_clusters_min1_count"], 2)
            self.assertEqual(
                batch["candidate_clusters_min1"],
                [[3.0, 4.0], [5.0, 6.0]],
            )
            scan_log = next(
                message for message in log_messages if "scan batch" in message)
            self.assertIn("label=room_LF_scan", scan_log)
            self.assertIn("raw=4", scan_log)
            self.assertIn("min_points=1", scan_log)
            self.assertIn("(3.0, 4.0)", scan_log)
            self.assertIn("(5.0, 6.0)", scan_log)
            self.assertEqual(list(Path(directory).glob("*.tmp.*")), [])

    def test_scan_batch_clustering_keeps_distinct_spheres_and_rejects_singletons(self):
        mod = _load_module()
        points = [(1.0, 2.0), (1.1, 2.05), (0.95, 1.98), (2.7, 2.0)]
        clustered = mod.cluster_scan_points(points, merge_radius=1.2, min_points=2)
        self.assertEqual(len(clustered), 1)
        self.assertAlmostEqual(clustered[0][0], 1.0, delta=0.1)

    def test_scan_batch_does_not_merge_simultaneous_candidates(self):
        mod = _load_module()
        points = [
            (1.0, 2.0, 1), (1.05, 2.0, 2), (1.1, 2.0, 3),
            (1.6, 2.0, 1), (1.65, 2.0, 2), (1.7, 2.0, 3),
        ]
        clustered = mod.cluster_scan_points(points, merge_radius=1.2, min_points=3)
        self.assertEqual(len(clustered), 2)

    def test_nearest_pose_uses_image_timestamp(self):
        mod = _load_module()
        poses = [(1.0, (1.0, 2.0, 0.6), 0.1), (1.2, (3.0, 4.0, 0.6), 0.3)]
        xyz, yaw = mod.nearest_pose(poses, 1.18, ((0.0, 0.0, 0.6), 0.0))
        self.assertEqual(xyz, (3.0, 4.0, 0.6))
        self.assertAlmostEqual(yaw, 0.3)

    def test_rgbd_depth_is_fused_and_background_return_is_rejected(self):
        mod = _load_module()
        depth = mod.select_sphere_depth(6.0, radius_px=55.0, fx=550.0)
        self.assertAlmostEqual(depth, 1.5, delta=0.05)
        consistent = mod.select_sphere_depth(1.55, radius_px=55.0, fx=550.0)
        self.assertAlmostEqual(consistent, 1.55)
        fallback = mod.select_sphere_depth(float("nan"), radius_px=55.0, fx=550.0)
        self.assertAlmostEqual(fallback, 1.5, delta=0.05)

    def test_recording_camera_planar_range_calibration_is_truth_independent(self):
        mod = _load_module()
        corrected = mod.scale_planar_range((4.0, -2.0), (0.0, 2.0, 0.6), 0.95)
        self.assertAlmostEqual(corrected[0], 3.8)
        self.assertAlmostEqual(corrected[1], -1.8)

    def test_disagreeing_near_horizon_ground_ray_prefers_sphere_depth(self):
        mod = _load_module()
        fused = mod.fuse_sphere_world_point(
            (3.0, 4.0, 0.2), (3.0, 7.0), target_z=0.15)
        self.assertEqual(fused, (3.0, 4.0, 0.15))
        agreeing = mod.fuse_sphere_world_point(
            (3.0, 4.0, 0.2), (3.2, 4.2), target_z=0.15)
        self.assertAlmostEqual(agreeing[0], 3.1)
        self.assertAlmostEqual(agreeing[1], 4.1)

    def test_room_door_halfplane_rejects_cross_room_projection(self):
        mod = _load_module()
        self.assertTrue(mod.point_inside_door_halfplane(
            (2.0, 10.0, 0.15), 1.1, 1.0))
        self.assertFalse(mod.point_inside_door_halfplane(
            (-1.2, 10.0, 0.15), 1.1, 1.0))
        self.assertTrue(mod.point_inside_door_halfplane(
            (-2.0, 10.0, 0.15), -1.1, -1.0))

    def test_detector_updates_only_during_scan_windows(self):
        mod = _load_module()
        self.assertTrue(mod.should_update_detections(True))
        self.assertFalse(mod.should_update_detections(False))

    def test_scan_min_points_allows_sparse_front_left_scan_only(self):
        mod = _load_module()
        self.assertEqual(mod.scan_min_points("room_LF_scan"), 1)
        self.assertEqual(mod.scan_min_points("room_LB_front_scan"), 3)
        self.assertEqual(mod.scan_min_points("room_RB_scan"), 3)

    def test_detects_red_circle_but_not_red_square_or_green_circle(self):
        mod = _load_module()
        image = np.zeros((800, 800, 3), np.uint8)
        cv2.circle(image, (500, 400), 45, (255, 0, 0), -1)
        cv2.rectangle(image, (120, 120), (210, 210), (255, 0, 0), -1)
        pentagon = np.array([[300, 100], [348, 135], [330, 192], [270, 192], [252, 135]], np.int32)
        cv2.fillPoly(image, [pentagon], (255, 0, 0))
        cv2.circle(image, (650, 650), 45, (0, 255, 0), -1)

        detections = mod.detect_red_spheres(image)

        self.assertEqual(len(detections), 1)
        self.assertAlmostEqual(detections[0][0], 500, delta=2)
        self.assertAlmostEqual(detections[0][1], 400, delta=2)

    def test_detects_partially_occluded_red_circle(self):
        mod = _load_module()
        image = np.zeros((200, 200, 3), np.uint8)
        cv2.circle(image, (100, 100), 40, (255, 0, 0), -1)
        # Simulate a wall/furniture edge hiding the upper part of the ball.
        cv2.rectangle(image, (0, 0), (200, 90), (0, 0, 0), -1)
        detections = mod.detect_red_spheres(image)
        self.assertEqual(len(detections), 1)

    def test_center_pixel_projects_forward_with_front_camera_pitch(self):
        mod = _load_module()
        calibration = mod.CameraCalibration.front_camera_fallback()

        point = mod.pixel_to_world(
            calibration.cx,
            calibration.cy,
            [0.0, 2.0, 0.6],
            math.pi / 2.0,
            calibration,
        )

        self.assertIsNotNone(point)
        self.assertAlmostEqual(point[0], 0.0, delta=0.05)
        self.assertAlmostEqual(point[1], 3.45, delta=0.12)

    def test_realsense_depth_projects_center_pixel_in_front_of_robot(self):
        mod = _load_module()
        calibration = mod.CameraCalibration.real_sense_fallback()

        point = mod.depth_pixel_to_world(
            calibration.cx,
            calibration.cy,
            2.0,
            [0.0, 2.0, 0.6],
            math.pi / 2.0,
            calibration,
        )

        self.assertAlmostEqual(point[0], 0.0, delta=0.05)
        self.assertAlmostEqual(point[1], 4.28, delta=0.08)
        self.assertAlmostEqual(point[2], 0.643, delta=0.02)

    def test_recording_camera_ground_projection_uses_monocular_geometry(self):
        mod = _load_module()
        calibration = mod.CameraCalibration.recording_camera_fallback()
        point = mod.pixel_to_world(
            calibration.cx,
            calibration.cy + 90.0,
            [0.0, 2.0, 0.31],
            math.pi / 2.0,
            calibration,
        )
        self.assertIsNotNone(point)
        self.assertAlmostEqual(point[0], 0.0, delta=0.05)
        self.assertGreater(point[1], 2.5)

        upper = mod.pixel_to_world(
            calibration.cx,
            calibration.cy + 90.0,
            [0.0, 2.0, 5.51],
            math.pi / 2.0,
            calibration,
            target_z=5.35,
        )
        self.assertIsNotNone(upper)
        self.assertAlmostEqual(upper[0], point[0], delta=0.05)
        self.assertAlmostEqual(upper[1], point[1], delta=0.05)

    def test_tracker_confirms_repeated_observations_without_merging_distinct_source(self):
        mod = _load_module()
        tracker = mod.DetectionTracker(merge_radius=0.5, confirm_count=3)

        tracker.update([(1.0, 2.0)])
        tracker.update([(1.1, 2.0)])
        tracker.update([(0.95, 2.05)])
        tracker.update([(2.4, 2.0)])

        confirmed = tracker.confirmed()
        self.assertEqual(len(confirmed), 1)
        self.assertAlmostEqual(confirmed[0][0], 1.0, delta=0.1)
        self.assertEqual(len(tracker.tracks), 2)

    def test_default_tracker_merges_projection_drift_but_keeps_distinct_sources(self):
        mod = _load_module()
        tracker = mod.DetectionTracker(confirm_count=1)
        tracker.update([(1.0, 2.0)])
        tracker.update([(1.7, 2.0)])
        tracker.update([(2.8, 2.0)])
        self.assertEqual(len(tracker.tracks), 2)

    def test_one_point_two_meter_tracker_keeps_adjacent_real_spheres_distinct(self):
        mod = _load_module()
        tracker = mod.DetectionTracker(merge_radius=1.2, confirm_count=1)
        tracker.update([(0.0, 0.0)])
        tracker.update([(1.45, 0.0)])
        self.assertEqual(len(tracker.tracks), 2)

    def _rescan_room_batches(self):
        """floor_1_room_2 of official seed 20: G3 blind, G4 scanned twice."""
        def diagnostic(radius_px):
            return {"accepted": True, "radius_px": radius_px,
                    "depth_center_m": 6.0, "depth_shape_accepted": False}

        return [
            {"room_id": "room_a", "viewpoint_role": "G3",
             "candidate_frames": 0, "raw_candidate_points": 0,
             "candidate_clusters_min1": [], "contour_diagnostics": []},
            {"room_id": "room_a", "viewpoint_role": "G4",
             "candidate_frames": 7, "raw_candidate_points": 11,
             "candidate_clusters_min1": [[-7.0101, 23.1025],
                                         [-5.0306, 23.7469],
                                         [-5.4438, 21.7848]],
             "contour_diagnostics": [diagnostic(5.50), diagnostic(6.74),
                                     diagnostic(5.24)]},
            {"room_id": "room_a", "viewpoint_role": "G4",
             "candidate_frames": 8, "raw_candidate_points": 11,
             "candidate_clusters_min1": [[-7.2317, 22.5156],
                                         [-5.0627, 23.4457]],
             "contour_diagnostics": [diagnostic(5.60), diagnostic(6.80)]},
        ]

    def test_second_room_scan_reappearance_keeps_small_single_view_track(self):
        mod = _load_module()
        batches = self._rescan_room_batches()
        for position in ([-7.0101, 23.1025, 2.75], [-5.0520, 23.5461, 2.75]):
            event = {"room_id": "room_a", "evidence_frames": 3,
                     "position": position}
            self.assertEqual(
                mod.independent_scan_batch_reappearances(event, batches), 2)
            self.assertFalse(
                mod.is_weak_small_single_view_detection(event, batches))

    def test_single_batch_small_single_view_track_is_still_rejected(self):
        mod = _load_module()
        batches = self._rescan_room_batches()
        event = {"room_id": "room_a", "evidence_frames": 3,
                 "position": [-5.4438, 21.7848, 2.75]}
        self.assertEqual(
            mod.independent_scan_batch_reappearances(event, batches), 1)
        self.assertTrue(
            mod.is_weak_small_single_view_detection(event, batches))

    def test_tracker_never_merges_same_xy_across_floors(self):
        mod = _load_module()
        tracker = mod.DetectionTracker(merge_radius=0.75, confirm_count=1)
        tracker.update([(1.0, 2.0, 0.15)])
        tracker.update([(1.0, 2.0, 2.75)])
        self.assertEqual(len(tracker.tracks), 2)
        self.assertEqual(len(tracker.confirmed()), 2)


if __name__ == "__main__":
    unittest.main()
