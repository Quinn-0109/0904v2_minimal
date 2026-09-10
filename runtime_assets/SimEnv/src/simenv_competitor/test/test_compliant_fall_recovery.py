#!/usr/bin/env python3
"""ROS-light tests for explicit multi-floor fall recovery safety."""

import math
import os
import sys
import threading
import types
import unittest

SCRIPT_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, SCRIPT_DIR)


class Duration:
    def __init__(self, value):
        self.value = float(value)

    def to_sec(self):
        return self.value


class Time:
    current = 10.0

    def __init__(self, value=0.0):
        self.value = float(value)

    @classmethod
    def now(cls):
        return cls(cls.current)

    def __sub__(self, other):
        return Duration(self.value - other.value)


class Quaternion:
    def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0):
        self.x, self.y, self.z, self.w = x, y, z, w


class Vector:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z


class Pose:
    def __init__(self):
        self.position = Vector()
        self.orientation = Quaternion()


class ModelState:
    def __init__(self):
        self.model_name = ""
        self.reference_frame = ""
        self.pose = Pose()


class Joy:
    def __init__(self):
        self.header = types.SimpleNamespace(stamp=None)
        self.axes, self.buttons = [], []


class Bool:
    def __init__(self, data=False):
        self.data = bool(data)


class String:
    def __init__(self, data=""):
        self.data = str(data)


def module(name, **values):
    result = types.ModuleType(name)
    for key, value in values.items():
        setattr(result, key, value)
    sys.modules[name] = result
    return result


rospy = module(
    "rospy", Time=Time, Duration=Duration, ServiceException=Exception,
    loginfo=lambda *args: None, logwarn=lambda *args: None,
    logerr_throttle=lambda *args: None, set_param=lambda *args: None)
gazebo = module("gazebo_msgs")
gazebo.msg = module(
    "gazebo_msgs.msg", ModelState=ModelState,
    ModelStates=type("ModelStates", (), {}))
gazebo.srv = module(
    "gazebo_msgs.srv", SetModelState=type("SetModelState", (), {}))
geometry = module("geometry_msgs")
geometry.msg = module("geometry_msgs.msg", Quaternion=Quaternion)
nav = module("nav_msgs")
nav.msg = module("nav_msgs.msg", Odometry=type("Odometry", (), {}))
sensor = module("sensor_msgs")
sensor.msg = module(
    "sensor_msgs.msg", Imu=type("Imu", (), {}), Joy=Joy)
std = module("std_msgs")
std.msg = module("std_msgs.msg", Bool=Bool, String=String)

from compliant_fall_recovery import (  # noqa: E402
    CompliantFallRecovery,
    floor_base_from_state,
    recovery_truth_sample_is_stable,
    relative_fall_evidence,
    stair_transition_is_active,
)


class Recorder:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class StairOwnershipTopicTest(unittest.TestCase):
    def test_resident_pretrigger_states_do_not_claim_stair_ownership(self):
        for phase in ("WAIT_F1", "WAIT_F2", "WAIT_F3"):
            self.assertFalse(stair_transition_is_active(phase), phase)

    def test_passive_home_recovery_wait_does_not_block_recovery_worker(self):
        self.assertFalse(stair_transition_is_active(
            "FIRST_FLOOR_HOME_FALL_RECOVERY_WAIT"))

    def test_roundtrip_descent_topic_is_owned_by_stair_controller(self):
        self.assertIn(
            "/simenv/third_to_first_floor_stair_state",
            CompliantFallRecovery._STAIR_TOPICS)
        self.assertTrue(stair_transition_is_active(
            "STAIR_DESCENT_FLIGHT_B"))
        self.assertTrue(stair_transition_is_active(
            "FIRST_FLOOR_HOME_RETURN"))

    def test_physical_stair_and_landing_states_keep_ownership(self):
        for phase in (
                "STAIR_ASCENT", "STAIR_ASCENT_B",
                "STAIR_LANDING_TURN", "SECOND_FLOOR_REACHED",
                "STAIR_DESCENT_FLIGHT_A", "STAIR_DESCENT_FLIGHT_B"):
            self.assertTrue(stair_transition_is_active(phase), phase)


def quaternion(roll=0.0, pitch=0.0, yaw=0.0):
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return Quaternion(
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy)


def pose(z, roll=0.0, x=1.0, y=2.0):
    result = Pose()
    result.position = Vector(x, y, z)
    result.orientation = quaternion(roll=roll)
    return result


def twist(linear=0.0, angular=0.0):
    return types.SimpleNamespace(
        linear=Vector(linear, 0.0, 0.0),
        angular=Vector(0.0, 0.0, angular))


def odom(z):
    return types.SimpleNamespace(
        pose=types.SimpleNamespace(pose=pose(z)))


def node():
    result = CompliantFallRecovery.__new__(CompliantFallRecovery)
    result._lock = threading.RLock()
    result._pose = None
    result._imu = types.SimpleNamespace(orientation=Quaternion())
    result._truth = pose(0.30)
    result._truth_twist = twist()
    result._truth_received_at = Time.now()
    result._truth_sample_sequence = 1
    result._first_floor_base = 0.0
    result._floor_height = 2.6
    result._floor_base = 0.0
    result._healthy_truth_z = None
    result._stair_active_by_topic = {
        topic: False for topic in CompliantFallRecovery._STAIR_TOPICS}
    result._roll_limit = 0.65
    result._pitch_limit = 0.65
    result._z_limit = 0.25
    result._flat_low_margin = 0.10
    result._upright_z = 0.55
    result._standing_body_height = 0.30
    result._minimum_healthy_relative_z = 0.27
    result._truth_maximum_age = 0.50
    result._recovery_height_tolerance = 0.08
    result._recovery_release_tilt = 0.35
    result._recovery_release_linear_speed = 0.12
    result._recovery_release_angular_speed = 0.25
    result._recovery_release_confirm_count = 2
    result._recovery_minimum_rl_seconds = 2.0
    result._recovery_maximum_duration = 12.0
    result._recovery_retry_limit = 0
    result._recovery_retry_count = 0
    result._recovering = False
    result._recovery_stuck = False
    result._recovery_floor_base = None
    result._recovery_expected_healthy_z = None
    result._recovery_release_streak = 0
    result._recovery_last_truth_sequence = -1
    result._recovery_started_at = None
    result._recover_step = 0
    result._recover_step_time = Time(0.0)
    result._candidate_floor_base = None
    result._candidate_expected_healthy_z = None
    result._candidate_last_truth_sequence = -1
    result._candidate_clear_streak = 0
    result._candidate_clear_confirm_count = 2
    result._fall_streak = 0
    result._confirm_count = 3
    result._mapping_pause_asserted = False
    result._start_grace = 0.0
    result._node_start = Time(0.0)
    result._recovery_count = 0
    result._cooldown = 0.0
    result._last_recover = Time(-100.0)
    result._use_gazebo_reset = False
    result._recovery_state_pub = Recorder()
    result._hold_pub = Recorder()
    result._joy_pub = Recorder()
    result._status_pub = Recorder()
    return result


class FallRecoveryTest(unittest.TestCase):
    def setUp(self):
        Time.current = 10.0

    def test_floor_translation_invariant_side_and_flat_detection(self):
        test_node = node()
        for base in (0.0, 2.6, 5.2):
            test_node._floor_base = base
            test_node._truth = pose(base + 0.10, roll=1.0)
            test_node._imu.orientation = quaternion(roll=1.0)
            self.assertTrue(test_node._is_fallen())
            test_node._truth = pose(base + 0.05)
            test_node._imu.orientation = Quaternion()
            self.assertTrue(test_node._is_fallen())
            test_node._truth = pose(base + 0.30)
            self.assertFalse(test_node._is_fallen())

    def test_phase_switches_base_and_clears_latched_stair_owner(self):
        self.assertEqual(floor_base_from_state(
            "/simenv/second_floor_state",
            "SECOND_FLOOR_CORRIDOR_ENTRY_REACHED", 2.6), 2.6)
        self.assertEqual(floor_base_from_state(
            "/simenv/third_floor_state",
            "THIRD_FLOOR_EXPLORATION_START", 2.6), 5.2)
        self.assertTrue(stair_transition_is_active("SECOND_FLOOR_REACHED"))
        test_node = node()
        test_node._stair_active_by_topic[
            "/simenv/stair_transition_state"] = True
        test_node._on_floor_state(
            "/simenv/second_floor_state",
            String("SECOND_FLOOR_EXPLORATION_START"))
        self.assertEqual(test_node._floor_base, 2.6)
        self.assertFalse(test_node._stair_active_by_topic[
            "/simenv/stair_transition_state"])

    def test_stair_activity_disables_flat_floor_watchdog(self):
        test_node = node()
        test_node._truth = pose(0.05, roll=1.0)
        test_node._imu.orientation = quaternion(roll=1.0)
        test_node._on_stair_state(
            "/simenv/stair_transition_state", String("STAIR_ASCENT"))
        self.assertFalse(test_node._is_fallen())

    def test_resident_descent_wait_does_not_disable_flat_floor_watchdog(self):
        test_node = node()
        test_node._truth = pose(0.05, roll=1.0)
        test_node._imu.orientation = quaternion(roll=1.0)
        test_node._on_stair_state(
            "/simenv/third_to_first_floor_stair_state", String("WAIT_F3"))
        self.assertTrue(test_node._is_fallen())

    def test_no_truth_uses_floor_local_odometry(self):
        test_node = node()
        test_node._floor_base = 2.6
        test_node._truth = None
        test_node._pose = odom(2.66)
        self.assertTrue(test_node._is_fallen())

    def test_f2_reset_target_does_not_drop_to_f1(self):
        test_node = node()
        test_node._floor_base = 2.6
        test_node._recovery_floor_base = 2.6
        test_node._truth = pose(2.66)
        test_node._truth_received_at = Time.now()
        captured = []
        test_node._set_model_state = lambda state: (
            captured.append(state) or types.SimpleNamespace(success=True))
        self.assertTrue(test_node._upright_in_place())
        self.assertAlmostEqual(captured[0].pose.position.z, 3.15)

    def test_gazebo_recovery_preserves_current_xy_and_healthy_yaw(self):
        test_node = node()
        test_node._use_gazebo_reset = True
        test_node._truth = pose(0.12, x=2.27, y=28.14)
        test_node._truth.orientation = quaternion(yaw=0.90)
        test_node._truth_received_at = Time.now()
        test_node._healthy_truth_anchor = (
            (1.66, 28.20, 0.31), -3.03)
        captured_states = []
        test_node._set_model_state = lambda state: (
            captured_states.append(state) or
            types.SimpleNamespace(success=True))
        captured_params = {}
        original_set_param = rospy.set_param
        rospy.set_param = lambda key, value: captured_params.__setitem__(
            key, value)
        try:
            self.assertTrue(test_node._begin_recovery())
        finally:
            rospy.set_param = original_set_param
        self.assertAlmostEqual(captured_params["/simenv/local_reset/x"], 2.27)
        self.assertAlmostEqual(captured_params["/simenv/local_reset/y"], 28.14)
        self.assertAlmostEqual(captured_params["/simenv/local_reset/yaw"], -3.03)
        self.assertAlmostEqual(captured_states[0].pose.position.x, 2.27)
        self.assertAlmostEqual(captured_states[0].pose.position.y, 28.14)
        actual_yaw = math.atan2(
            2.0 * captured_states[0].pose.orientation.w *
            captured_states[0].pose.orientation.z,
            1.0 - 2.0 * captured_states[0].pose.orientation.z ** 2)
        self.assertAlmostEqual(actual_yaw, -3.03)

    def test_first_suspect_sample_holds_cmd_and_map(self):
        test_node = node()
        test_node._truth = pose(0.05)
        test_node._on_timer(None)
        self.assertTrue(test_node._mapping_pause_asserted)
        self.assertTrue(test_node._hold_pub.messages[-1].data)
        self.assertTrue(test_node._recovery_state_pub.messages[-1].data)
        self.assertFalse(test_node._recovering)

    def test_candidate_clear_requires_truth_height_pose_and_speed(self):
        test_node = node()
        test_node._truth = pose(0.05)
        test_node._on_timer(None)
        test_node._truth = pose(0.30)
        test_node._truth_twist = twist(linear=0.5)
        test_node._truth_sample_sequence = 2
        test_node._truth_received_at = Time.now()
        test_node._on_timer(None)
        self.assertTrue(test_node._mapping_pause_asserted)

        for sequence in (3, 4):
            Time.current += 0.1
            test_node._truth_twist = twist()
            test_node._truth_sample_sequence = sequence
            test_node._truth_received_at = Time.now()
            test_node._on_timer(None)
        self.assertFalse(test_node._mapping_pause_asserted)
        self.assertFalse(test_node._hold_pub.messages[-1].data)

    def test_minimum_relative_height_blocks_release(self):
        common = dict(
            expected_z=2.90, roll=0.0, pitch=0.0,
            linear_speed=0.0, angular_speed=0.0,
            height_tolerance=0.08, tilt_limit=0.35,
            linear_speed_limit=0.12, angular_speed_limit=0.25,
            minimum_relative_z=0.27)
        self.assertFalse(recovery_truth_sample_is_stable(
            z=2.85, relative_z=0.25, **common))
        self.assertTrue(recovery_truth_sample_is_stable(
            z=2.90, relative_z=0.30, **common))

    def test_recovery_releases_only_after_consecutive_stable_samples(self):
        test_node = node()
        test_node._recovering = True
        test_node._mapping_pause_asserted = True
        test_node._recover_step = 2
        test_node._recovery_started_at = Time(0.0)
        test_node._recover_step_time = Time(5.0)
        test_node._recovery_floor_base = 2.6
        test_node._recovery_expected_healthy_z = 2.90
        test_node._truth = pose(2.90)
        test_node._recovery_last_truth_sequence = 1
        for sequence in (2, 3):
            test_node._truth_sample_sequence = sequence
            test_node._truth_received_at = Time.now()
            test_node._advance_recovery(Time.now())
        self.assertFalse(test_node._recovering)
        self.assertFalse(test_node._mapping_pause_asserted)
        self.assertFalse(test_node._hold_pub.messages[-1].data)

    def test_timeout_retries_once_before_fail_closed(self):
        test_node = node()
        test_node._recovering = True
        test_node._mapping_pause_asserted = True
        test_node._recovery_retry_limit = 1
        test_node._recover_step = 2
        test_node._recovery_started_at = Time(0.0)
        test_node._recover_step_time = Time(5.0)
        test_node._advance_recovery(Time(12.1))
        self.assertFalse(test_node._recovery_stuck)
        self.assertEqual(test_node._recovery_retry_count, 1)
        self.assertEqual(test_node._recover_step, 0)
        self.assertEqual(test_node._status_pub.messages[-1].data,
                         "RECOVERY_RETRY_1")
        self.assertEqual(test_node._joy_pub.messages[-1].buttons[1], 1)
        test_node._advance_recovery(Time(24.2))
        self.assertTrue(test_node._recovery_stuck)
        self.assertEqual(test_node._status_pub.messages[-1].data,
                         "RECOVERY_FAILED_STUCK")


    def test_timeout_is_fail_closed_and_reports_stuck(self):
        test_node = node()
        test_node._recovering = True
        test_node._mapping_pause_asserted = True
        test_node._hold_pub.publish(Bool(data=True))
        test_node._recover_step = 2
        test_node._recovery_started_at = Time(0.0)
        test_node._recover_step_time = Time(5.0)
        test_node._advance_recovery(Time(12.1))
        self.assertTrue(test_node._recovering)
        self.assertTrue(test_node._recovery_stuck)
        self.assertTrue(test_node._mapping_pause_asserted)
        self.assertTrue(test_node._hold_pub.messages[-1].data)
        self.assertEqual(
            test_node._status_pub.messages[-1].data,
            "RECOVERY_FAILED_STUCK")


if __name__ == "__main__":
    unittest.main()
