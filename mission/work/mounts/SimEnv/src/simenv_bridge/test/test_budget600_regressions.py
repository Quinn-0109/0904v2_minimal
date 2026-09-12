"""Focused tests: no ROS master, simulation or generated scene required."""
import json
import math
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from test_three_floor_rl_mission import load_sequencer, load_supervisor


class Budget600RegressionTest(unittest.TestCase):
    def setUp(self):
        self.mod = load_sequencer()
        self.seq = self.mod.ThreeFloorGoalSequencer.__new__(
            self.mod.ThreeFloorGoalSequencer)
        self.seq._lock = threading.RLock()
        self.seq._failure = None

    def test_alternating_drift_cannot_keep_watchdog_alive(self):
        update = self.mod.advance_progress_anchors
        distance, heading, _ = update(1., 1., math.inf, math.inf, .08)
        distance, heading, progress = update(.9, 1.3, distance, heading, .08)
        self.assertTrue(progress)
        self.assertEqual(heading, 1.)
        for _ in range(20):
            for d, h in ((1.2, 1.), (.9, 1.3)):
                distance, heading, progress = update(d, h, distance, heading, .08)
                self.assertFalse(progress)
        self.assertFalse(update(.9, math.inf, .9, math.inf, .08)[2])
        self.assertTrue(update(.81, math.inf, .9, math.inf, .08)[2])
        self.assertTrue(update(.9, .90, .9, 1., .08)[2])

    def test_blocked_translation_cannot_be_extended_by_heading_progress(self):
        deadline = self.mod.direct_progress_deadline
        # Fresh yaw progress at t=100 does not extend a blocked position whose
        # last positional advance was t=80. Original code waited until 108.
        self.assertEqual(deadline(100., 80., 15., 8., True, .25, .12), 88.)
        self.assertEqual(deadline(100., 80., 15., 8., False, .25, .12), 115.)
        self.assertEqual(deadline(100., 80., 15., 8., True, .10, .12), 115.)
        # Actual position progress or a bounded recovery starts a fresh window.
        self.assertEqual(deadline(100., 100., 15., 8., True, .25, .12), 108.)

    def test_failure_blocks_late_return_stage_and_running_timing(self):
        mod = load_supervisor()
        obj = mod.ThreeFloorRLMissionSupervisor.__new__(mod.ThreeFloorRLMissionSupervisor)
        obj._lock = threading.RLock()
        obj._write_lock = threading.RLock()
        obj._failure = 'exploration_deadline_exceeded_600.0s'
        obj._completed = False
        obj._task_stage = None
        obj._start_exploration_clock = lambda: self.fail('terminal clock restarted')
        obj._transition_task_stage('return_to_first_floor_lobby', 'late callback')
        self.assertIsNone(obj._task_stage)
        obj._exploration_started_monotonic = 1.
        obj._exploration_started_sim_elapsed = 0.
        obj._sim_clock_elapsed = 600.356
        obj._started_monotonic = 0.
        obj._exploration_started_wall_time = 1.
        obj._sim_clock_resets = 0
        obj._task_stages = []
        obj._exploration_start_pose = None
        obj.task_budget = 600.
        obj.timing_config = {}
        with tempfile.TemporaryDirectory() as directory:
            obj.timing_path = str(Path(directory) / 'timing.json')
            obj._write_timing('running')
            result = json.loads(Path(obj.timing_path).read_text())
        self.assertEqual(result['status'], 'failed')
        self.assertFalse(result['budget_met'])

    def test_final_yaw_gain_changes_only_stationary_alignment(self):
        for command in (self.mod.direct_rl_command,
                        self.mod.direct_holonomic_rl_command):
            args = dict(maximum_yaw_rate=1.5, target_yaw=.5,
                        position_tolerance=.12, heading_tolerance=.16)
            moving = command((0, 0, 0, 0), (2, 0), .9, **args)
            self.assertEqual(moving, command((0, 0, 0, 0), (2, 0), .9,
                                            final_yaw_gain=1.8, **args))
            turn = command((0, 0, 0, 0), (0, 0), .9,
                           final_yaw_gain=1.8, **args)
            self.assertEqual(turn[0], 0.)
            self.assertAlmostEqual(turn[-3], .9)
            self.assertFalse(turn[-1])
            args['target_yaw'] = 3.
            self.assertEqual(command((0, 0, 0, 0), (0, 0), .9,
                                     final_yaw_gain=1.8, **args)[-3], 1.5)
            args['target_yaw'] = .15
            self.assertTrue(command((0, 0, 0, 0), (0, 0), .9,
                                    final_yaw_gain=1.8, **args)[-1])

    def test_policy_dedup_keeps_ack_refresh(self):
        seq = self.seq
        seq._plane_policy = '/tmp/policy_act_inference_plane.pt'
        seq._stair_policy = '/tmp/policy_act_inference_stair.pt'
        seq._policy_ack_at = {}
        events = []
        seq._record_event = lambda *a, **kw: events.append((a, kw))
        msg = SimpleNamespace(data='policy_reloaded:' + seq._plane_policy)
        with patch.object(self.mod.time, 'monotonic', side_effect=[1., 2.]):
            seq._on_policy_status(msg)
            seq._on_policy_status(msg)
        self.assertEqual(seq._policy_ack_at['plane'], 2.)
        self.assertEqual(len(events), 1)

    def test_event_journal_survives_ring_eviction(self):
        seq = self.seq
        seq._events = []
        self.mod.rospy.get_time = lambda: 12.5
        with tempfile.TemporaryDirectory() as directory:
            seq._output_dir = directory
            for i in range(1002):
                seq._record_event('test', index=i)
            lines = (Path(directory) / 'route_events.jsonl').read_text().splitlines()
            self.assertEqual(len(lines), 1002)
            self.assertEqual(len(seq._events), 1000)
            self.assertEqual(json.loads(lines[0])['sim_time'], 12.5)

    def test_changing_numeric_telemetry_is_throttled_but_guard_is_not(self):
        seq = self.seq
        seq._plane_policy = 'plane.pt'
        seq._stair_policy = 'stair.pt'
        events = []
        seq._record_event = lambda *a, **kw: events.append(kw)
        with patch.object(self.mod.time, 'monotonic', side_effect=[1., 2., 3., 8.]):
            for timestamp, hold in ((1, False), (2, False), (3, True), (8, True)):
                seq._on_policy_status(SimpleNamespace(data=json.dumps(dict(
                    timestamp=timestamp, phase='LOCOMOTION_READY',
                    plane_tilt_guard_hold=hold, command_vx=timestamp))))
        self.assertEqual(len(events), 3)

    def test_geometry_skip_keeps_real_error(self):
        seq = self.seq
        seq._record_event = lambda *a, **kw: None
        waypoint = dict(x=1., y=0., room_id='room', note='g4')
        for geometry_ok, accepted_short in ((True, False), (False, True)):
            seq._actual_oblique_waypoint_ok = lambda *a: geometry_ok
            seq._last_leg_accepted_short = accepted_short
            pose, error, _, corrected = seq._correct_oblique_waypoint(
                {}, waypoint, (.5, 0., 0., 0.), .9, 1)
            self.assertEqual(error, .5)
            self.assertFalse(corrected)

    def test_near_target_alone_cannot_bypass_geometry(self):
        seq = self.seq
        seq._actual_oblique_waypoint_ok = lambda *a: False
        seq._last_leg_accepted_short = False
        seq._viewpoint_stall_accept = .6
        seq._record_event = lambda *a, **kw: None
        seq._room_record = lambda *a: {'geometry_correction_count': 0}
        seq._drive_direct_waypoint = lambda *a: ((.5, 0., 0., 0.), .5, 0)
        result = seq._correct_oblique_waypoint(
            {'floor_number': 1}, dict(x=1., y=0., note='g4'),
            (.5, 0., 0., 0.), .9, 1)
        self.assertIsNone(result[0])
        self.assertIn('actual_oblique_geometry_failed', seq._failure)

    def test_blocked_accept_still_requires_sensor_and_distance(self):
        seq = self.seq
        seq._viewpoint_block_evidence_hits = 2
        seq._viewpoint_stall_accept = .6
        waypoint = dict(room_id='r', room_phase='G4')
        self.assertTrue(seq._viewpoint_blocked_short(waypoint, .5, 2, .5))
        for distance, best, hits in ((.5, .5, 1), (.7, .5, 2),
                                     (.5, .7, 2), (math.inf, .5, 2)):
            self.assertFalse(seq._viewpoint_blocked_short(
                waypoint, distance, hits, best))

    def test_repeated_cloud_is_not_independent_obstacle_evidence(self):
        seq = self.seq
        seq._latest_cloud_monotonic = 10.
        seq._target_has_sensed_obstacle = lambda *a: True
        stamp, hit = seq._new_target_obstacle_sample({}, (), (), None)
        self.assertTrue(hit)
        stamp, hit = seq._new_target_obstacle_sample({}, (), (), stamp)
        self.assertFalse(hit)
        seq._latest_cloud_monotonic = 11.
        stamp, hit = seq._new_target_obstacle_sample({}, (), (), stamp)
        self.assertTrue(hit)
        seq._latest_cloud_monotonic = 12.
        seq._target_has_sensed_obstacle = lambda *a: False
        _, hit = seq._new_target_obstacle_sample({}, (), (), stamp)
        self.assertFalse(hit)


if __name__ == '__main__':
    unittest.main()
