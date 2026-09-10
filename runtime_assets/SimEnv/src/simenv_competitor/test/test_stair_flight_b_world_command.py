#!/usr/bin/env python3
import math
import os
import sys
import unittest
from types import SimpleNamespace


SCRIPTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

from stair_transition_manager import StairTransition


class FlightBWorldCommandTest(unittest.TestCase):
    def manager(self, x=-2.485, yaw=-math.pi / 2.0):
        manager = StairTransition.__new__(StairTransition)
        manager.truth_pose = [x, 5.0, 4.2, yaw]
        manager.truth_flight_b_heading = -math.pi / 2.0
        manager.truth_flight_b_pose = [-2.485, 5.0, 4.2, yaw]
        manager.truth_flight_b_entry_x = -2.485
        manager.truth_flight_b_recovery_active = False
        manager.truth_flight_b_recovery_release_heading_error = 0.08
        manager.truth_flight_b_recovery_heading_error = 0.18
        manager.truth_flight_b_recovery_forward_speed = 0.08
        manager.truth_flight_b_recovery_yaw_rate = 0.34
        manager.truth_flight_b_center_deadband = 0.04
        manager.truth_flight_b_center_speed = 0.12
        manager.truth_flight_b_center_gain = 0.55
        manager.truth_flight_b_max_yaw_rate = 0.10
        manager.truth_flight_b_heading_gain = 0.35
        manager.ascent_speed = 0.80
        return manager

    def test_forward_axis_is_world_negative_y(self):
        vx, vy, _, center_error, _ = self.manager()._truth_flight_b_control()
        self.assertAlmostEqual(center_error, 0.0, places=6)
        self.assertAlmostEqual(vx, 0.0, places=6)
        self.assertAlmostEqual(vy, -0.80, places=6)

    def test_centerline_correction_only_changes_world_x(self):
        vx, vy, _, center_error, _ = self.manager(x=-4.0)._truth_flight_b_control()
        self.assertGreater(center_error, 0.0)
        self.assertAlmostEqual(vx, 0.12, places=6)
        self.assertAlmostEqual(vy, -0.80, places=6)

    def high_water_manager(self, x=-2.48, y=2.52, z=5.40):
        manager = self.manager(x=x)
        manager.source_floor_index = 1
        manager.truth_pose = [x, y, z, -math.pi / 2.0]
        manager.truth_f2_f3_atomic_final_tread_handoff = True
        manager.truth_f2_f3_atomic_high_water_max_y = 2.85
        manager.truth_f2_f3_atomic_high_water_center_tolerance = 1.0
        manager.truth_flight_b_top = [-2.485, 1.80, 5.42, 0.0]
        manager.total_height_gain = 2.40
        manager.truth_upper_landing_height_shortfall_tolerance = 0.10
        return manager

    def test_high_water_capture_requires_legacy_atomic_mode(self):
        manager = self.high_water_manager()
        self.assertTrue(
            manager._truth_f2_f3_atomic_high_water_ready(2.49))
        manager.truth_f2_f3_atomic_final_tread_handoff = False
        self.assertFalse(
            manager._truth_f2_f3_atomic_high_water_ready(2.49))

    def test_high_water_capture_rejects_low_or_off_center_pose(self):
        low = self.high_water_manager(z=5.20)
        off_center = self.high_water_manager(x=-1.20)
        before_final_tread = self.high_water_manager(y=3.10)

        self.assertFalse(low._truth_f2_f3_atomic_high_water_ready(2.29))
        self.assertFalse(
            off_center._truth_f2_f3_atomic_high_water_ready(2.49))
        self.assertFalse(
            before_final_tread._truth_f2_f3_atomic_high_water_ready(2.49))

    def physical_landing_manager(self):
        manager = self.high_water_manager(y=1.25, z=5.42)
        manager.truth_flight_b_pose = [-2.485, 5.0, 4.2, -math.pi / 2.0]
        manager.truth_flight_b_next_pose = [-2.485, 4.7]
        manager.truth_second_floor_clearance = 0.30
        manager.truth_f2_f3_landing_recovery_margin = 0.30
        manager.truth_upper_landing_center_tolerance = 0.35
        manager.truth_upper_landing_heading_tolerance = 0.35
        manager.truth_upper_landing_linear_speed = 0.20
        manager.truth_upper_landing_angular_speed = 0.35
        manager.truth_f1_f2_landing_tilt_tolerance = 0.30
        manager.truth_model_pose = SimpleNamespace(orientation=SimpleNamespace(
            x=0.0, y=0.0, z=math.sin(-math.pi / 4.0),
            w=math.cos(-math.pi / 4.0)))
        manager.truth_model_twist = SimpleNamespace(
            linear=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            angular=SimpleNamespace(x=0.0, y=0.0, z=0.0))
        return manager

    def test_physical_landing_requires_interior_clearance_and_quiet_posture(self):
        manager = self.physical_landing_manager()
        self.assertTrue(manager._truth_f2_f3_physical_landing_ready(2.40))

        manager.truth_pose[1] = 1.60
        self.assertFalse(manager._truth_f2_f3_physical_landing_ready(2.40))
        manager.truth_pose[1] = 1.25
        manager.truth_model_twist.linear.z = 0.30
        self.assertFalse(manager._truth_f2_f3_physical_landing_ready(2.40))
        manager.truth_model_twist.linear.z = 0.0
        manager.truth_model_pose.orientation.x = math.sin(0.20)
        self.assertFalse(manager._truth_f2_f3_physical_landing_ready(2.40))

    def controller_manager(self):
        manager = self.manager()
        manager.source_floor_index = 1
        manager.truth_f2_f3_physical_only_recovery = True
        manager.locomotion_ever_ready = False
        manager.locomotion_ready = False
        manager.locomotion_lost_since = None
        manager.truth_f2_f3_controller_loss_timeout = 0.50
        return manager

    def test_controller_loss_requires_prior_ready_and_physical_f2_f3(self):
        manager = self.controller_manager()
        self.assertFalse(manager._truth_f2_f3_controller_lost(12.0))
        manager.locomotion_ever_ready = True
        manager.locomotion_lost_since = 10.0
        manager.locomotion_ready = True
        self.assertFalse(manager._truth_f2_f3_controller_lost(12.0))
        manager.locomotion_ready = False
        manager.source_floor_index = 0
        self.assertFalse(manager._truth_f2_f3_controller_lost(12.0))
        manager.source_floor_index = 1
        manager.truth_f2_f3_physical_only_recovery = False
        self.assertFalse(manager._truth_f2_f3_controller_lost(12.0))

    def test_controller_loss_uses_bounded_confirmation_dwell(self):
        manager = self.controller_manager()
        manager.locomotion_ever_ready = True
        manager.locomotion_lost_since = 10.0
        self.assertFalse(manager._truth_f2_f3_controller_lost(10.49))
        self.assertTrue(manager._truth_f2_f3_controller_lost(10.50))

    def test_ready_callback_clears_controller_loss_timer(self):
        manager = self.controller_manager()
        manager.locomotion_ever_ready = True
        manager.locomotion_lost_since = 10.0
        manager.on_locomotion_ready(SimpleNamespace(data=True))
        self.assertTrue(manager.locomotion_ready)
        self.assertIsNone(manager.locomotion_lost_since)

    def pre_riser_manager(self, source_floor_index=0, x=-4.02,
                          yaw=math.pi / 2.0):
        manager = self.manager(x=x, yaw=yaw)
        manager.source_floor_index = source_floor_index
        manager.truth_entry_guide = True
        manager.truth_pose = [x, 1.75, 0.31, yaw]
        manager.truth_step_pose = [-4.02, 2.17, 0.31, yaw]
        manager.truth_stair_heading = math.pi / 2.0
        manager.truth_f1_pre_riser_rebased = False
        return manager

    def test_force_flag_accepts_already_physical_f1_pre_riser(self):
        manager = self.pre_riser_manager()
        self.assertTrue(manager._rebase_f1_pre_riser_to_truth(force=True))
        self.assertTrue(manager.truth_f1_pre_riser_rebased)

    def test_physical_pre_riser_rejects_far_pose_without_teleport(self):
        manager = self.pre_riser_manager(x=-3.50)
        self.assertFalse(manager._rebase_f1_pre_riser_to_truth(force=True))
        self.assertAlmostEqual(manager.truth_pose[0], -3.50)


if __name__ == "__main__":
    unittest.main()
