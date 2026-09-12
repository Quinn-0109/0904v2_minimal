"""Recorded G4 positions; observer is scan-end odometry, not image-time truth.

This checks the existing rule's response to supplied observer evidence, not
an exact replay of the run, whose output omitted observer_xy.
"""
import unittest
from test_danger_detector import _load_module


class ObserverEvidenceTest(unittest.TestCase):
    def test_recorded_pair_is_suppressed_when_observer_is_available(self):
        mod = _load_module()
        records = [dict(floor=2, room_id='floor_1_room_3', viewpoint_role='G4',
                        waypoint='floor_1_room_3_g4', observer_xy=[3.9322, 28.2032],
                        position=position)
                   for position in ([2.9097, 25.0171, 2.75], [2.5922, 24.1599, 2.75],
                                    [8.6681, 32.2875, 2.75])]
        self.assertEqual(mod.same_ray_duplicate_losers(records), {1})
        for record in records:
            record.pop('observer_xy')
        self.assertEqual(mod.same_ray_duplicate_losers(records), set())

    def test_output_exposes_missing_observer_without_inventing_a_pose(self):
        from test_merge_position_ordering import run_write_output, track, event
        metadata = event(1, 'G4', 'room_g4', (3.9322, 28.2032), 1.)
        metadata.pop('observer_xy')
        result = run_write_output([track(1, 2.9097, 25.0171, 10)], [metadata])
        self.assertEqual(result['tracks_missing_observer_xy'], 1)
        self.assertEqual(result['detector_rules_revision'],
                         'same_ray_before_merge_midpoint_after_dedup_v1')
        self.assertEqual(result['confirmed_count'], 1)


if __name__ == '__main__':
    unittest.main()
