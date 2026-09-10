#!/usr/bin/env python3
import inspect
import os
import sys
import threading
import unittest
from collections import deque
from types import SimpleNamespace
from unittest import mock


SCRIPT_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'scripts'))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from stair_descent_manager import StairDescent


class _Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message.data)


class StairDescentTriggerTest(unittest.TestCase):
    def _stand_capture_manager(self):
        manager = StairDescent.__new__(StairDescent)
        manager._joint_lock = threading.RLock()
        manager.joint_names = list(manager._FSM_JOINT_ORDER)
        manager.joint_positions = [0.0, .9, -1.8] * 4
        manager.joint_velocity_rms = .08
        manager.joint_received_at = 10.0
        manager.policy_switch_joint_velocity_rms = .35
        manager.policy_switch_snapshot_window = .60
        manager.policy_switch_snapshot_min_samples = 8
        manager.policy_switch_snapshot_started = 9.4
        manager._stand_target_param = '/stand_target_joints'
        manager._stand_target_applied = False
        manager.stand_target_evidence = None
        manager.joint_samples = deque(maxlen=512)
        return manager

    def test_fixed_stand_target_requires_a_stable_window_not_one_frame(self):
        manager = self._stand_capture_manager()
        manager.joint_samples.append((
            10.0, dict(zip(manager._FSM_JOINT_ORDER,
                           manager.joint_positions)), .08))

        with mock.patch('stair_descent_manager.time.monotonic',
                        return_value=10.0), \
                mock.patch('stair_descent_manager.rospy.set_param') as setter:
            self.assertFalse(manager._capture_stand_target())

        setter.assert_not_called()

    def test_fixed_stand_window_is_projected_to_symmetric_support(self):
        manager = self._stand_capture_manager()
        # Deliberately asymmetric low-speed gait frames: locking the final
        # frame directly reproduces run176's unsafe single-support target.
        for index in range(9):
            stamp = 9.4 + index * .075
            phase = .18 if index % 2 else -.18
            pose = [
                .12, .82 + phase, -1.68 - phase,
                -.10, .98 - phase, -1.92 + phase,
                .08, .85 - phase, -1.72 + phase,
                -.11, .95 + phase, -1.88 - phase,
            ]
            manager.joint_samples.append((
                stamp, dict(zip(manager._FSM_JOINT_ORDER, pose)), .08))

        with mock.patch('stair_descent_manager.time.monotonic',
                        return_value=10.0), \
                mock.patch('stair_descent_manager.rospy.set_param') as setter, \
                mock.patch('stair_descent_manager.rospy.loginfo'):
            self.assertTrue(manager._capture_stand_target())

        target = setter.call_args.args[1]
        self.assertEqual(target[0::3], [0.0] * 4)
        self.assertEqual(len(set(round(v, 6) for v in target[1::3])), 1)
        self.assertEqual(len(set(round(v, 6) for v in target[2::3])), 1)
        self.assertEqual(manager.stand_target_evidence['sample_count'], 9)
        self.assertEqual(manager.stand_target_evidence['source'],
                         'stable_window_symmetric_support')

    def _landing_manager(self, pose, attitude, segment=0):
        manager = StairDescent.__new__(StairDescent)
        manager.truth_pose = pose
        manager.truth_attitude = attitude
        manager.start_floor_number = 3
        manager.segment = segment
        manager.flight_a_end_y = 2.40
        manager.landing_maximum_tilt = 0.14
        manager.second_floor_landing_maximum_z = 3.00
        manager.first_floor_landing_maximum_z = 0.48
        manager._flight_a_center_x = mock.Mock(return_value=-3.60)
        return manager

    def _failed_manager_at_height(self, height):
        manager = StairDescent.__new__(StairDescent)
        manager.phase = 'WAIT_F3'
        manager.mission_failure_tokens = {'THIRD_FLOOR_EXPLORATION_FAILED'}
        manager.mission_trigger_tokens = set()
        manager.terminate_on_mission_failure = False
        manager.truth_pose = (0.0, 0.0, height, 0.0)
        manager.truth_attitude = (0.0, 0.0, 0.0)
        manager.home_minimum_upright_z = 0.20
        manager.home_maximum_tilt = 0.65
        manager.minimum_third_floor_failure_wait_height = 4.80
        manager.first_floor_failure_return_maximum_z = 1.20
        manager.second_floor_failure_return_minimum_z = 2.30
        manager.second_floor_failure_return_maximum_z = 3.40
        manager._terminal = mock.Mock()
        return manager

    def test_landing_cross_rate_uses_effective_stair_translation_floor(self):
        manager = StairDescent.__new__(StairDescent)
        manager.landing_position_tolerance = 0.20
        manager.landing_recenter_speed = 0.40
        manager.landing_minimum_x_speed = 0.30
        manager.landing_position_gain = 0.90

        self.assertEqual(manager._landing_cross_rate(0.199), 0.0)
        self.assertAlmostEqual(manager._landing_cross_rate(0.203), 0.30)
        self.assertAlmostEqual(manager._landing_cross_rate(-0.203), -0.30)

    def test_partial_exploration_failure_waits_for_physical_return(self):
        manager = StairDescent.__new__(StairDescent)
        manager.phase = 'WAIT_F3'
        manager.mission_failure_tokens = {'THIRD_FLOOR_EXPLORATION_FAILED'}
        manager.mission_trigger_tokens = {
            'THIRD_FLOOR_STAIR_RETURN_READY',
            'THIRD_FLOOR_STAIR_RETURN_PARTIAL_READY'}
        manager.pending_mission_failure_token = None
        manager.pending_mission_failure_wall_time = None
        manager.state = _Publisher()

        with mock.patch('stair_descent_manager.rospy.logwarn'):
            manager.on_mission_trigger(SimpleNamespace(
                data='THIRD_FLOOR_EXPLORATION_FAILED'))

        self.assertEqual(manager.phase, 'WAIT_F3')
        self.assertEqual(manager.pending_mission_failure_token,
                         'THIRD_FLOOR_EXPLORATION_FAILED')
        self.assertEqual(manager.state.messages[-1],
                         'WAITING_FOR_BOUNDED_THIRD_FLOOR_STAIR_RETURN')

    def test_run63_failed_f3_at_landing_starts_bounded_descent(self):
        manager = self._failed_manager_at_height(5.52)
        manager.truth_pose = (-0.92, 1.74, 5.52, -2.75)
        manager.truth_landing_center = (-0.90, 1.55)
        manager.third_floor_failure_landing_trigger_radius = 1.25
        manager.pending_mission_failure_token = None
        manager.pending_mission_failure_wall_time = None
        manager.state = _Publisher()
        manager._begin_segment = mock.Mock(
            side_effect=lambda: setattr(
                manager, 'phase', 'STAIR_DESCENT_ENTRY_GUIDE'))
        manager._record_trace = mock.Mock()

        with mock.patch('stair_descent_manager.rospy.logwarn'), \
                mock.patch('stair_descent_manager.rospy.Time') as ros_time:
            ros_time.now.return_value.to_sec.return_value = 1312.3
            manager.on_mission_trigger(SimpleNamespace(
                data='THIRD_FLOOR_EXPLORATION_FAILED'))

        manager._begin_segment.assert_called_once_with()
        self.assertEqual(manager.phase, 'STAIR_DESCENT_ENTRY_GUIDE')
        self.assertEqual(manager.state.messages[-1],
                         'STAIR_DESCENT_ENTRY_GUIDE')
        manager._terminal.assert_not_called()

    def test_failed_f3_partial_return_wait_has_its_own_bounded_timeout(self):
        manager = StairDescent.__new__(StairDescent)
        manager.phase = 'WAIT_F3'
        manager.started = None
        manager.pending_mission_failure_wall_time = 100.0
        manager.mission_failure_return_wait_timeout = 45.0
        manager._pause_active = False
        manager._pause_pub = _Publisher()
        manager._boot_monotonic = 0.0
        manager.trigger_timeout = 2400.0
        manager._terminal = mock.Mock()

        with mock.patch('stair_descent_manager.time.time', return_value=145.1), \
                mock.patch('stair_descent_manager.time.monotonic',
                           return_value=145.1):
            manager.tick(None)

        manager._terminal.assert_called_once_with(
            'STAIR_DESCENT_NOT_TRIGGERED_F3_RETURN_TIMEOUT',
            'f3_failure_bounded_stair_return_token_timeout', error=True)

    def _waiting_trigger_manager(self):
        manager = StairDescent.__new__(StairDescent)
        manager.phase = 'WAIT_F3'
        manager.started = None
        manager.pending_mission_failure_token = None
        manager.pending_mission_failure_wall_time = None
        manager._pause_active = False
        manager._pause_pub = _Publisher()
        manager._boot_monotonic = 0.0
        manager._boot_ros_time = 100.0
        manager.trigger_timeout = 2400.0
        manager.trigger_wall_timeout = 14400.0
        manager._terminal = mock.Mock()
        manager._poll_persisted_mission_failure = mock.Mock()
        return manager

    def test_low_rtf_wall_elapsed_does_not_expire_ros_trigger_budget(self):
        manager = self._waiting_trigger_manager()
        with mock.patch('stair_descent_manager.rospy.Time') as ros_time, \
                mock.patch('stair_descent_manager.time.monotonic',
                           return_value=3000.0), \
                mock.patch('stair_descent_manager.time.sleep'):
            ros_time.now.return_value.to_sec.return_value = 1400.0
            manager.tick(None)
        manager._terminal.assert_not_called()

    def test_trigger_budget_expires_in_ros_simulation_time(self):
        manager = self._waiting_trigger_manager()
        with mock.patch('stair_descent_manager.rospy.Time') as ros_time, \
                mock.patch('stair_descent_manager.time.monotonic',
                           return_value=3000.0), \
                mock.patch('stair_descent_manager.time.sleep'):
            ros_time.now.return_value.to_sec.return_value = 2500.1
            manager.tick(None)
        manager._terminal.assert_called_once_with(
            'STAIR_DESCENT_TRIGGER_TIMEOUT',
            'stair_descent_trigger_ros_timeout', error=True)

    def test_pending_failure_routes_home_when_truth_pose_arrives(self):
        manager = self._failed_manager_at_height(2.91)
        manager.started = None
        manager.pending_mission_failure_token = (
            'THIRD_FLOOR_EXPLORATION_FAILED')
        manager.pending_mission_failure_wall_time = 100.0
        manager._pause_active = False
        manager._pause_pub = _Publisher()
        manager._begin_second_floor_failure_return = mock.Mock()

        manager.tick(None)

        manager._begin_second_floor_failure_return.assert_called_once_with(
            'THIRD_FLOOR_EXPLORATION_FAILED')
        self.assertIsNone(manager.pending_mission_failure_token)

    def test_persisted_failure_recovers_missed_short_lived_topic(self):
        manager = self._failed_manager_at_height(2.91)
        manager.out = '/tmp/run37'
        manager.persisted_failure_poll_wall_time = 0.0
        manager.pending_mission_failure_token = None
        manager.pending_mission_failure_wall_time = None
        manager._begin_second_floor_failure_return = mock.Mock()
        payload = '{"state": "THIRD_FLOOR_EXPLORATION_FAILED"}'

        with mock.patch('stair_descent_manager.time.time',
                        return_value=101.0), \
                mock.patch('builtins.open',
                           mock.mock_open(read_data=payload)), \
                mock.patch('stair_descent_manager.rospy.logwarn'):
            manager._poll_persisted_mission_failure()

        manager._begin_second_floor_failure_return.assert_called_once_with(
            'THIRD_FLOOR_EXPLORATION_FAILED')

    def test_first_floor_upstream_failure_returns_home_without_credit(self):
        manager = self._failed_manager_at_height(0.45)
        manager._begin_home_return = mock.Mock()

        with mock.patch('stair_descent_manager.rospy.logwarn'), \
                mock.patch('stair_descent_manager.rospy.Time') as ros_time:
            ros_time.now.return_value.to_sec.return_value = 12.5
            manager.on_mission_trigger(SimpleNamespace(
                data='THIRD_FLOOR_EXPLORATION_FAILED'))

        manager._begin_home_return.assert_called_once_with()
        manager._terminal.assert_not_called()
        self.assertEqual(
            manager.lower_floor_failure_return_mode,
            'F1_HOME_AFTER_THIRD_FLOOR_EXPLORATION_FAILED')

    def test_first_floor_room_failure_uses_corridor_then_home_route(self):
        manager = StairDescent.__new__(StairDescent)
        manager.first_floor_landing_ros_time = None
        manager.lower_floor_failure_return_mode = (
            'F1_HOME_AFTER_FIRST_FLOOR_EXPLORATION_FAILED')
        manager.truth_pose = (-1.70, 15.67, 0.55, 0.0)
        manager.home_failure_corridor_x = 0.0
        manager.home_failure_corridor_return_y = 1.55
        manager.return_to_start_enabled = True
        manager.plane_policy = '/tmp/policy_plane.pt'
        manager.home_x = 0.0
        manager.home_y = 0.0
        manager.home_failure_egress_max_recoveries = 1
        manager.pub = _Publisher()
        manager.state = _Publisher()
        manager.cmd = mock.Mock()

        with mock.patch('stair_descent_manager.rospy.Time') as ros_time, \
                mock.patch('stair_descent_manager.time.monotonic',
                           return_value=50.0), \
                mock.patch('stair_descent_manager.rospy.loginfo'):
            ros_time.now.return_value.to_sec.return_value = 120.0
            manager._begin_home_return()

        self.assertEqual(manager.home_stage, 'failure_room_egress')
        self.assertEqual(manager._home_return_target(), (0.0, 15.67))
        manager.home_stage = 'failure_corridor_return'
        self.assertEqual(manager._home_return_target(), (0.0, 1.55))
        manager.home_stage = 'home'
        self.assertEqual(manager._home_return_target(), (0.0, 0.0))
        self.assertEqual(manager.phase, 'FIRST_FLOOR_HOME_POLICY_LOADING')

    def test_second_floor_upstream_failure_starts_bounded_descent(self):
        manager = self._failed_manager_at_height(2.91)
        manager._begin_second_floor_failure_return = mock.Mock()

        manager.on_mission_trigger(SimpleNamespace(
            data='THIRD_FLOOR_EXPLORATION_FAILED'))

        manager._begin_second_floor_failure_return.assert_called_once_with(
            'THIRD_FLOOR_EXPLORATION_FAILED')
        manager._terminal.assert_not_called()

    def test_interfloor_upstream_failure_remains_explicit(self):
        manager = self._failed_manager_at_height(1.75)

        with mock.patch('stair_descent_manager.rospy.logerr'):
            manager.on_mission_trigger(SimpleNamespace(
                data='THIRD_FLOOR_EXPLORATION_FAILED'))

        manager._terminal.assert_called_once_with(
            'STAIR_DESCENT_NOT_TRIGGERED_SOURCE_FLOOR_FAILED',
            'stair_descent_not_triggered_source_floor_failed', error=True)

    def test_second_floor_failure_return_reuses_physical_descent_chain(self):
        manager = StairDescent.__new__(StairDescent)
        manager.plane_policy = '/tmp/policy_plane.pt'
        manager._geometry_dirty = False
        manager.truth_entry_guide = False
        manager.skip_entry_guide = True
        manager.truth_landing_center = (9.0, 9.0)
        manager._load_landing_center = mock.Mock()
        manager.state = mock.Mock()
        manager.cmd = mock.Mock()

        with mock.patch('stair_descent_manager.rospy.logwarn'), \
                mock.patch('stair_descent_manager.rospy.Time') as ros_time:
            ros_time.now.return_value.to_sec.return_value = 22.0
            manager._begin_second_floor_failure_return(
                'THIRD_FLOOR_EXPLORATION_FAILED')

        self.assertEqual(manager.start_floor_number, 2)
        self.assertEqual(manager.end_floor_number, 1)
        self.assertEqual(manager.total_segments, 1)
        self.assertTrue(manager.truth_entry_guide)
        self.assertFalse(manager.skip_entry_guide)
        self.assertEqual(manager.phase,
                         'SECOND_FLOOR_FAILURE_RETURN_POLICY_LOADING')
        self.assertFalse(manager.policy_loaded)
        self.assertEqual(manager.active_policy, '/tmp/policy_plane.pt')
        self.assertIn('F2_TO_F1_AFTER_',
                      manager.lower_floor_failure_return_mode)

    def _stair_policy_loading_manager(self):
        manager = StairDescent.__new__(StairDescent)
        manager.phase = 'STAIR_DESCENT_ENTRY_GUIDE'
        manager.policy = '/tmp/policy_stair.pt'
        manager.plane_policy = '/tmp/policy_plane.pt'
        manager.active_policy = manager.plane_policy
        # Reproduce run56: the preceding plane-policy transaction left this
        # generic flag set when the physical entry guide reached the stair.
        manager.policy_requested = True
        manager.policy_loaded = True
        manager.stair_policy_loading_started_at = None
        manager.stair_policy_loading_last_request = None
        manager.stair_policy_loading_request_count = 0
        manager.stair_policy_loading_retry_period = 2.0
        manager.stair_policy_loading_timeout = 12.0
        manager.stair_policy_loading_max_requests = 4
        manager.policy_warmup_seconds = 1.2
        manager.pub = _Publisher()
        manager.state = _Publisher()
        manager.cmd = mock.Mock()
        manager.hold_rl = mock.Mock()
        manager._record_trace = mock.Mock()
        manager._terminal = mock.Mock()
        return manager

    def test_stair_policy_request_is_not_suppressed_by_plane_request_flag(self):
        manager = self._stair_policy_loading_manager()

        with mock.patch('stair_descent_manager.time.monotonic',
                        return_value=100.0):
            manager._enter_policy_loading()

        # Policy transfer now deliberately starts with a physical settle and
        # FixedStand transaction.  The stale generic request flag must not
        # skip that safety gate, and the queued-policy branch must overwrite
        # it unconditionally when the fresh stand acknowledgement arrives.
        self.assertEqual(manager.phase, 'STAIR_DESCENT_POLICY_SETTLE')
        self.assertEqual(manager.active_policy, '/tmp/policy_plane.pt')
        self.assertTrue(manager.policy_requested)
        self.assertEqual(manager.pub.messages, [])
        queued_source = inspect.getsource(
            StairDescent._policy_switch_stand_tick)
        self.assertIn('self.active_policy=self.policy', queued_source)
        self.assertIn('self.policy_loaded=False', queued_source)
        self.assertIn('self.policy_requested=True', queued_source)
        self.assertIn('self.pub.publish(String(data=self.policy))',
                      queued_source)

    def test_stair_policy_loading_retries_and_times_out_boundedly(self):
        manager = self._stair_policy_loading_manager()
        manager.phase = 'STAIR_DESCENT_POLICY_LOADING'
        manager.active_policy = manager.policy
        manager.policy_loaded = False
        manager.stair_policy_loading_started_at = 100.0
        manager.stair_policy_loading_last_request = 100.0
        manager.stair_policy_loading_request_count = 1

        with mock.patch('stair_descent_manager.time.monotonic',
                        return_value=102.1):
            manager._policy_loading_tick()
        self.assertEqual(manager.pub.messages, ['/tmp/policy_stair.pt'])
        self.assertEqual(manager.stair_policy_loading_request_count, 2)
        manager._terminal.assert_not_called()

        with mock.patch('stair_descent_manager.time.monotonic',
                        return_value=112.1):
            manager._policy_loading_tick()
        manager._terminal.assert_called_once_with(
            'STAIR_DESCENT_POLICY_LOADING_TIMEOUT',
            'stair_descent_policy_loading_timeout', error=True)

    def test_first_descent_fall_reanchors_once_then_remains_bounded(self):
        manager = StairDescent.__new__(StairDescent)
        manager.truth_fall_recovery_enabled = True
        manager.truth_fall_recovery_max_attempts = 1
        manager.flight_fall_recoveries = 0
        manager.fall_recovery_count = 0
        manager.flight_b_descent_drop = 1.05
        manager.flight_a_descent_drop = 2.45
        manager.fall_window_seconds = 0.8
        manager.truth_pose = (-3.1, 3.2, 2.2, 1.57)
        manager.side_slip_guard_window_seconds = 6.0
        manager._correct_no_drop_stair_seam = mock.Mock(return_value=True)

        with mock.patch('stair_descent_manager.time.monotonic',
                        return_value=100.0):
            self.assertTrue(manager._recover_flight_fall('b', 0.95))

        manager._correct_no_drop_stair_seam.assert_called_once_with(
            'b', tread_index=4, reason='fall_recovery')
        self.assertEqual(manager.flight_fall_recoveries, 1)
        self.assertEqual(manager.fall_recovery_count, 1)
        self.assertFalse(manager._recover_flight_fall('b', 0.95))

    def test_f2_final_tread_is_not_accepted_as_flat_landing(self):
        # run31 reached the final tread with this still-inclined pose.  It must
        # continue forward onto the slab before the 180 degree segment turn.
        manager = self._landing_manager(
            (-3.548, 2.175, 3.058, -1.478), (-0.132, 0.354))

        self.assertFalse(manager._target_floor_landing_envelope())

    def test_f2_flat_slab_pose_is_accepted(self):
        # Replayed from successful run28.
        manager = self._landing_manager(
            (-3.608, 1.740, 2.923, -1.520), (0.003, 0.027))

        self.assertTrue(manager._target_floor_landing_envelope())

    def test_f1_inclined_pose_is_not_accepted_as_flat_landing(self):
        manager = self._landing_manager(
            (-3.55, 2.10, 0.58, -1.50), (0.02, 0.20), segment=1)

        self.assertFalse(manager._target_floor_landing_envelope())

    def test_run32_flat_landing_is_inside_segment_turn_band(self):
        source_path = os.path.join(SCRIPT_DIR, 'stair_descent_manager.py')
        with open(source_path, 'r', encoding='utf-8') as stream:
            source = stream.read()

        self.assertIn(
            "'~truth_descent_segment_turn_y_low_m', 1.45", source)
        self.assertLessEqual(1.45, 1.62)


if __name__ == '__main__':
    unittest.main()
