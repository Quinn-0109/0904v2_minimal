#!/usr/bin/env python3
"""ROS-light tests for anchored FAST-LIO/truth relative-pose comparison."""

import math
import os
import sys
import types
import threading
import unittest


SCRIPT_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "scripts"))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


try:
    import rospy  # noqa: F401
    from gazebo_msgs.msg import ModelStates  # noqa: F401
    from nav_msgs.msg import Odometry  # noqa: F401
    from std_msgs.msg import Bool, String  # noqa: F401
except ImportError:
    rospy_module = sys.modules.setdefault("rospy", types.ModuleType("rospy"))

    def install_message_stub(package, class_name):
        package_module = sys.modules.setdefault(
            package, types.ModuleType(package))
        message_name = package + ".msg"
        message_module = sys.modules.setdefault(
            message_name, types.ModuleType(message_name))
        message_class = type(class_name, (), {
            "__init__": lambda self, **kwargs: self.__dict__.update(kwargs),
        })
        setattr(message_module, class_name, message_class)
        setattr(package_module, "msg", message_module)

    install_message_stub("gazebo_msgs", "ModelStates")
    install_message_stub("nav_msgs", "Odometry")
    install_message_stub("std_msgs", "Bool")
    install_message_stub("std_msgs", "String")


from localization_truth_recovery import (  # noqa: E402
    LocalizationTruthRecovery,
    StableSe2RelativeMotionAnchor,
    relative_error_exceeds,
)


def transform_pose(pose, rotation, translation):
    cosine = math.cos(rotation)
    sine = math.sin(rotation)
    return (
        translation[0] + cosine * pose[0] - sine * pose[1],
        translation[1] + sine * pose[0] + cosine * pose[1],
        math.atan2(
            math.sin(pose[2] + rotation),
            math.cos(pose[2] + rotation)),
    )


class LocalizationTruthRecoverySe2Test(unittest.TestCase):
    @staticmethod
    def make_node():
        node = LocalizationTruthRecovery.__new__(LocalizationTruthRecovery)
        node._enabled = True
        node._minimum_planar_error_m = 0.75
        node._minimum_yaw_error_rad = 0.35
        node._quiet_speed_mps = 0.12
        node._quiet_angular_rps = 0.12
        node._registration_unhealthy_grace_sec = 0.35
        node._request_cooldown_sec = 2.0
        node._maximum_requests = 24
        node._last_request_at = -math.inf
        node._request_count = 0
        node._stair_phase_active = False
        node._fall_recovery_active = False
        node._fall_recovery_cleared_at = -math.inf
        node._post_fall_stability_sec = 0.40
        node._truth_upright = True
        node._last_fastlio_recovery_applied_count = 0
        node._lock = threading.RLock()
        node._registration = {"healthy": True}
        node._registration_unhealthy_since = None
        node._truth_twist = (0.0, 0.0, 0.0)
        node._relative_motion_anchor = StableSe2RelativeMotionAnchor(
            sync_tolerance_sec=0.10,
            stability_sec=0.0,
            sample_freshness_sec=0.25)
        return node

    def test_rotated_translated_frames_do_not_request_recovery(self):
        rotation = math.pi / 2.0
        translation = (40.0, -25.0)
        truth_anchor = (4.0, -3.0, 0.35)
        truth_current = (7.5, 1.25, 1.10)
        odom_anchor = transform_pose(truth_anchor, rotation, translation)
        odom_current = transform_pose(truth_current, rotation, translation)
        self.assertGreater(
            math.hypot(
                odom_anchor[0] - truth_anchor[0],
                odom_anchor[1] - truth_anchor[1]),
            20.0)

        node = self.make_node()
        node._truth_pose = (
            truth_anchor[0], truth_anchor[1], 0.3, truth_anchor[2])
        node._odom_pose = odom_anchor
        node._truth_received_at = 10.00
        node._odom_received_at = 10.02
        self.assertIsNone(node._should_request_recovery(10.03))
        self.assertTrue(node._relative_motion_anchor.anchored)

        node._truth_pose = (
            truth_current[0], truth_current[1], 0.3, truth_current[2])
        node._odom_pose = odom_current
        node._truth_received_at = 11.00
        node._odom_received_at = 11.02
        self.assertIsNone(node._should_request_recovery(11.03))

    def test_true_relative_drift_requests_recovery(self):
        rotation = math.pi / 2.0
        translation = (40.0, -25.0)
        truth_anchor = (4.0, -3.0, 0.35)
        truth_current = (7.5, 1.25, 1.10)

        node = self.make_node()
        node._truth_pose = (
            truth_anchor[0], truth_anchor[1], 0.3, truth_anchor[2])
        node._odom_pose = transform_pose(
            truth_anchor, rotation, translation)
        node._truth_received_at = 20.00
        node._odom_received_at = 20.01
        self.assertIsNone(node._should_request_recovery(20.02))

        expected_odom = transform_pose(
            truth_current, rotation, translation)
        node._truth_pose = (
            truth_current[0], truth_current[1], 0.3, truth_current[2])
        node._odom_pose = (
            expected_odom[0] + 0.90,
            expected_odom[1],
            expected_odom[2])
        node._truth_received_at = 21.00
        node._odom_received_at = 21.01
        payload = node._should_request_recovery(21.02)
        self.assertIsNotNone(payload)
        self.assertEqual(
            payload["reason"], "relative_pose_truth_disagreement")
        self.assertAlmostEqual(payload["planar_error_m"], 0.9, places=6)

    def test_anchor_requires_stable_fresh_synchronized_samples(self):
        monitor = StableSe2RelativeMotionAnchor(
            sync_tolerance_sec=0.10,
            stability_sec=0.40,
            sample_freshness_sec=0.30)
        odom_pose = (50.0, -20.0, math.pi / 2.0)
        truth_pose = (0.0, 0.0, 0.0)

        self.assertIsNone(monitor.observe(
            odom_pose, truth_pose, 1.00, 0.75, 1.02, stable=True))
        self.assertFalse(monitor.anchored)
        self.assertIsNone(monitor.observe(
            odom_pose, truth_pose, 2.00, 2.01, 2.02, stable=False))
        self.assertIsNone(monitor.observe(
            odom_pose, truth_pose, 3.00, 3.01, 3.02, stable=True))
        self.assertIsNone(monitor.observe(
            odom_pose, truth_pose, 3.20, 3.21, 3.22, stable=True))
        error = monitor.observe(
            odom_pose, truth_pose, 3.42, 3.43, 3.44, stable=True)
        self.assertTrue(monitor.anchored)
        self.assertAlmostEqual(error["planar_error_m"], 0.0, places=9)
        self.assertFalse(relative_error_exceeds(error, 0.75, 0.35))

    def test_fall_freezes_monitor_but_preserves_valid_prefall_anchor(self):
        node = self.make_node()
        node._relative_motion_anchor.observe(
            (10.0, 5.0, 0.2), (1.0, 2.0, -0.1),
            1.0, 1.0, 1.0, stable=True)
        self.assertTrue(node._relative_motion_anchor.anchored)

        node._on_fall_recovery(types.SimpleNamespace(data=True))
        node._truth_upright = False
        self.assertIsNone(node._should_request_recovery(2.0))
        self.assertTrue(node._relative_motion_anchor.anchored)

        node._on_fall_recovery(types.SimpleNamespace(data=False))
        self.assertTrue(node._relative_motion_anchor.anchored)

    def test_non_upright_truth_cannot_create_anchor(self):
        node = self.make_node()
        node._truth_upright = False
        node._truth_pose = (0.0, 0.0, 0.3, 0.0)
        node._odom_pose = (20.0, -10.0, 1.0)
        node._truth_received_at = 4.0
        node._odom_received_at = 4.0
        self.assertIsNone(node._should_request_recovery(4.01))
        self.assertFalse(node._relative_motion_anchor.anchored)

    def test_publishing_request_immediately_resets_monitor_anchor(self):
        node = self.make_node()
        node._relative_motion_anchor.observe(
            (10.0, 5.0, 0.2), (1.0, 2.0, -0.1),
            1.0, 1.0, 1.0, stable=True)
        published = []
        written = []
        node._recovery_pub = types.SimpleNamespace(publish=published.append)
        node._writer = types.SimpleNamespace(submit=written.append)
        node._publish_recovery({
            "reason": "relative_pose_truth_disagreement",
            "relative_se2_anchor": True,
        })
        self.assertFalse(node._relative_motion_anchor.anchored)
        self.assertEqual(len(published), 1)
        self.assertEqual(len(written), 1)


    def test_stair_completion_discards_previous_floor_anchor(self):
        node = self.make_node()
        node._relative_motion_anchor.observe(
            (10.0, 5.0, 0.2), (1.0, 2.0, -0.1),
            1.0, 1.0, 1.0, stable=True)
        node._on_stair_state(
            "/simenv/second_to_third_floor_stair_state",
            types.SimpleNamespace(data="STAIR_ASCENT_B"))
        self.assertTrue(node._stair_phase_active)
        self.assertTrue(node._relative_motion_anchor.anchored)
        node._on_stair_state(
            "/simenv/second_to_third_floor_stair_state",
            types.SimpleNamespace(data="THIRD_FLOOR_HANDOFF_COMPLETE"))
        self.assertFalse(node._stair_phase_active)
        self.assertFalse(node._relative_motion_anchor.anchored)


    def test_floor_rebase_and_applied_correction_reset_anchor(self):
        node = self.make_node()
        node._relative_motion_anchor.observe(
            (10.0, 5.0, 0.2), (1.0, 2.0, -0.1),
            1.0, 1.0, 1.0, stable=True)
        self.assertTrue(node._relative_motion_anchor.anchored)
        node._on_floor_state(
            "/simenv/second_floor_state",
            types.SimpleNamespace(
                data="SECOND_FLOOR_LOCALIZATION_STABILIZING"))
        self.assertFalse(node._relative_motion_anchor.anchored)

        node._relative_motion_anchor.observe(
            (11.0, 5.0, 0.2), (2.0, 2.0, -0.1),
            2.0, 2.0, 2.0, stable=True)
        self.assertTrue(node._relative_motion_anchor.anchored)
        node._on_fastlio_recovery_status(types.SimpleNamespace(
            data="{\"applied_count\":1,"
                 "\"decision\":\"applied_returning_to_fastlio\"}"))
        self.assertFalse(node._relative_motion_anchor.anchored)


    def test_relative_yaw_drift_crosses_se2_threshold(self):
        error = {
            "planar_error_m": 0.05,
            "yaw_error_rad": 0.40,
        }
        self.assertTrue(relative_error_exceeds(error, 0.75, 0.35))


if __name__ == "__main__":
    unittest.main()
