#!/usr/bin/env python3
"""Static integration contract for simulator-only FAST-LIO recovery."""

import os
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FAST_LIO = os.path.join(ROOT, "FAST_LIO")
LASER_MAPPING = os.path.join(FAST_LIO, "src", "laserMapping.cpp")
RUNTIME = os.path.join(FAST_LIO, "src", "truth_recovery_runtime.inc")
FALL_RUNTIME = os.path.join(
    FAST_LIO, "src", "fall_quarantine_runtime.inc")
FAST_LIO_CONFIG = os.path.join(FAST_LIO, "config", "simenv_mid360.yaml")
MONITOR = os.path.join(
    ROOT, "simenv_competitor", "scripts", "localization_truth_recovery.py")


def read(path):
    with open(path, "r", encoding="utf-8") as stream:
        return stream.read()


class FastlioTruthRecoveryContractTest(unittest.TestCase):
    def test_request_has_a_real_fastlio_consumer_and_status_ack(self):
        source = read(LASER_MAPPING)
        self.assertIn('"/simenv/fastlio_truth_recovery_request"', source)
        self.assertIn("truth_recovery_request_callback", source)
        self.assertIn(
            '"/simenv/second_to_third_floor_stair_state"', source)
        self.assertIn('"/simenv/fastlio_truth_recovery_status"', source)

    def test_recovery_runs_after_guards_and_before_publish_and_map_insert(self):
        source = read(LASER_MAPPING)
        divergence = source.index(
            "apply_anchored_truth_divergence_gate();")
        registration = source.index("apply_registration_guard();")
        stationary = source.index("apply_stationary_motion_guard();")
        recovery = source.index("apply_requested_truth_recovery_guard();")
        publish = source.index("publish_odometry(pubOdomAftMapped);")
        map_insert = source.index(
            "if (fall_quarantine_output_allowed) {")
        self.assertLess(divergence, registration)
        self.assertLess(registration, stationary)
        self.assertLess(stationary, recovery)
        self.assertLess(recovery, publish)
        self.assertLess(publish, map_insert)

    def test_one_shot_recovery_never_teleports_or_clears_the_map(self):
        runtime = read(RUNTIME)
        self.assertNotIn("\"/gazebo/set_model_state\"", runtime)
        self.assertNotIn("SetModelState", runtime)
        self.assertNotIn("ikdtree.Build", runtime)
        self.assertIn("registration_measurement_valid = false", runtime)
        self.assertIn("state_point.pos(0) = target.x", runtime)
        self.assertIn("state_point.pos(1) = target.y", runtime)
        self.assertNotIn(
            "state_point.pos(0) = second_floor_truth_position", runtime)
        self.assertNotIn(
            "state_point.pos(1) = second_floor_truth_position", runtime)
        self.assertIn("motion_guard_recent_linear_speed = 0.0", runtime)
        self.assertIn(
            "motion_guard_recent_motion_stamp = ros::Time(0)", runtime)

    def test_divergence_gate_is_bounded_and_does_not_widen_recovery(self):
        config = read(FAST_LIO_CONFIG)
        self.assertIn("maximum_planar_correction_m: 2.0", config)
        self.assertIn("divergence_planar_threshold_m: 1.25", config)
        self.assertIn(
            "divergence_self_recovery_planar_threshold_m: 0.35", config)
        self.assertIn("divergence_yaw_threshold_rad: 0.55", config)

    def test_truth_is_time_aligned_and_stationary_sync_is_conditional(self):
        source = read(LASER_MAPPING)
        runtime = read(RUNTIME)
        self.assertIn("time_aligned_truth_pose()", runtime)
        self.assertIn("extrapolate_world_pose", runtime)
        self.assertIn(
            "truth_age <= truth_recovery_anchor_maximum_age", runtime)
        self.assertIn("const bool state_rewritten", source)
        valid = source.index("if (!registration_measurement_valid ||")
        rewritten = source.index("const bool state_rewritten", valid)
        sync = source.index("registration_anchor = state_point;", rewritten)
        self.assertLess(valid, rewritten)
        self.assertLess(rewritten, sync)

    def test_raw_planar_rebase_is_disabled_in_runtime_configuration(self):
        config = read(FAST_LIO_CONFIG)
        self.assertIn("rebase_planar_pose: false", config)
        self.assertNotIn("rebase_planar_pose: true", config)

    def test_monitor_resets_by_floor_and_ack_and_freezes_during_fall(self):
        monitor = read(MONITOR)
        self.assertIn('"/simenv/fall_recovery_active"', monitor)
        self.assertIn('"/simenv/fastlio_truth_recovery_status"', monitor)
        self.assertIn('"LOCALIZATION_STABILIZING"', monitor)
        self.assertIn("not self._truth_upright", monitor)
        self.assertIn('"recovery_request_published"', monitor)

    def test_continuous_truth_preserves_proven_anchor_across_local_fall_reset(self):
        runtime = read(FALL_RUNTIME)
        self.assertIn(
            "if (!truth_recovery_continuous_planar_localization)", runtime)
        guarded_reset = runtime.index(
            "if (!truth_recovery_continuous_planar_localization)")
        anchor_reset = runtime.index(
            "second_floor_truth_anchor_valid = false", guarded_reset)
        frozen_capture = runtime.index(
            "fall_quarantine_frozen_state =", anchor_reset)
        self.assertLess(guarded_reset, anchor_reset)
        self.assertLess(anchor_reset, frozen_capture)


if __name__ == "__main__":
    unittest.main()
