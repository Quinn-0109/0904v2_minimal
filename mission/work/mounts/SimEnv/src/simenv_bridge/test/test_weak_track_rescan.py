import json
import unittest
from types import SimpleNamespace
from test_danger_detector import _load_module


class WeakTrackRescanTest(unittest.TestCase):
    def run_case(self, count):
        mod = _load_module()
        detector = mod.Detector.__new__(mod.Detector)
        detector.scan_active = False
        detector.scan_label = 'floor_1_room_0_scan_g4'
        detector.stage = 'floor_1_exploration'
        detector.floor = 1
        detector.room_id = 'floor_1_room_0'
        detector.viewpoint_role = 'G4'
        detector.scan_points = []
        detector.scan_frame = 0
        detector.scan_batches = []
        detector._scan_batch = None
        detector.scan_merge_radius = 1.2
        track = dict(id=1, x=2., y=3., count=count,
                     metadata={'room_id': detector.room_id})
        detector.tracker = SimpleNamespace(confirmed_tracks=lambda: [track])
        detector._danger_rescan_requested_rooms = set()
        detector._room_scan_evidence = {}
        messages = []
        detector.danger_rescan_pub = SimpleNamespace(
            publish=lambda msg: messages.append(json.loads(msg.data)))
        detector.String = lambda data: SimpleNamespace(data=data)
        detector.rospy = SimpleNamespace(loginfo=lambda *a: None)
        detector.terminal_status = None
        detector._write_output = lambda status: None
        for _ in range(2):
            detector._scan_active_cb(SimpleNamespace(data=True))
            detector.scan_points = [(2., 3., frame) for frame in range(3)]
            detector._scan_batch['candidate_frames'] = 3
            detector._scan_batch['contour_diagnostics'] = [
                dict(accepted=True, radius_px=6.)]
            detector._scan_active_cb(SimpleNamespace(data=False))
        # Retrying never edits track count or creates a confirmation.
        self.assertEqual(track['count'], count)
        return messages

    def test_weak_confirmed_track_requests_one_retry(self):
        messages = self.run_case(3)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]['weak_track_ids'], [1])

    def test_strong_confirmed_track_does_not_add_retry(self):
        self.assertEqual(self.run_case(5), [])


if __name__ == '__main__':
    unittest.main()
