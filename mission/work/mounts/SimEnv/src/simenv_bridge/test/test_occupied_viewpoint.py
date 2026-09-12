import copy
import math
import threading
import unittest

from test_three_floor_rl_mission import load_sequencer


class OccupiedViewpointTest(unittest.TestCase):
    def setUp(self):
        self.m = load_sequencer()
        self.s = self.m.ThreeFloorGoalSequencer.__new__(self.m.ThreeFloorGoalSequencer)
        self.s._lock = threading.RLock()
        self.s._pose = (-4.3, 14.15, 0., 0.)
        self.s._runtime_audit_clearance = .38
        self.points = [(-4.148, 15.545, .293), (-4.226, 15.547, .182)]
        self.s._perceived_obstacle_points = lambda *a: self.points
        self.record = {'viewpoints': {'G3': {'actual_pose': [-5.782, 13.5321, 0., 0.]}}}
        self.s._room_record = lambda *a: self.record
        self.s._runtime_3d_audit = lambda *a: True
        self.events = []
        self.s._record_event = lambda name, **kw: self.events.append((name, kw))
        self.floor = {'floor_number': 1, 'z': 0.}
        self.w = dict(x=-3.8825, y=15.5431666667, scan=True, room_phase='G4',
                      room_id='floor_0_room_0', note='floor_0_room_0_g4',
                      viewpoint_policy='open_middle_deep_then_near_oblique',
                      room_type='open', tolerance=.12, minimum_viewpoint_separation_m=2.2,
                      door_contract={'centre': [-1.1, 14.865], 'inward_direction': -1},
                      actual_geometry_limits=dict(g3_depth_minimum=4.1525,
                          g3_depth_maximum=5.285, g4_depth_minimum=2.6425,
                          g4_depth_maximum=3.775, minimum_depth_delta=1.,
                          minimum_lateral_delta=.8, central_half_width_m=4.209,
                          angle_minimum=20., angle_maximum=70.))

    def test_recorded_endpoint_blocks_detour_but_nearby_viewpoint_is_safe(self):
        old = (self.w['x'], self.w['y'])
        self.assertIsNone(self.m.detour_point_from_points(
            self.s._pose, old, self.points, .38, .55))
        self.assertTrue(self.s._relocate_occupied_g4(self.floor, self.w))
        new = (self.w['x'], self.w['y'])
        self.assertLessEqual(math.dist(old, new), .600001)
        self.assertTrue(self.s._actual_oblique_waypoint_ok(self.floor, self.w, new))
        for p in self.points:
            self.assertGreaterEqual(self.m.point_segment_distance_2d(p, self.s._pose, new), .50)
        self.assertEqual(self.w['runtime_original_target'], list(old))
        self.assertFalse(self.s._relocate_occupied_g4(self.floor, self.w))

    def test_missing_evidence_or_missing_first_view_does_not_move_target(self):
        for points, first in (([], self.record['viewpoints']), (self.points, {})):
            self.points = points
            self.record['viewpoints'] = first
            before = copy.deepcopy(self.w)
            self.assertFalse(self.s._relocate_occupied_g4(self.floor, self.w))
            self.assertEqual(self.w, before)

    def test_rejected_live_audit_and_impossible_geometry_do_not_move_target(self):
        before = copy.deepcopy(self.w)
        self.s._runtime_3d_audit = lambda *a: False
        self.assertFalse(self.s._relocate_occupied_g4(self.floor, self.w))
        self.assertEqual(self.w, before)
        self.s._runtime_3d_audit = lambda *a: True
        self.w['minimum_viewpoint_separation_m'] = 100.
        before = copy.deepcopy(self.w)
        self.assertFalse(self.s._relocate_occupied_g4(self.floor, self.w))
        self.assertEqual(self.w, before)

    def test_ordinary_transit_and_g3_are_not_relocated(self):
        for phase in ('G3', 'G4_PATH', 'EXIT'):
            self.w['room_phase'] = phase
            self.assertFalse(self.s._relocate_occupied_g4(self.floor, self.w))


if __name__ == '__main__':
    unittest.main()
