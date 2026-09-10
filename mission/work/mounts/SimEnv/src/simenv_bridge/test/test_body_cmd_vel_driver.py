#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import sys
import types
import unittest


def _install_ros_stubs():
    sys.modules["rospy"] = types.ModuleType("rospy")
    geometry_msgs_msg = types.ModuleType("geometry_msgs.msg")
    geometry_msgs_msg.Twist = type("Twist", (), {})
    gazebo_msgs_msg = types.ModuleType("gazebo_msgs.msg")
    gazebo_msgs_msg.ModelState = type("ModelState", (), {})
    gazebo_msgs_srv = types.ModuleType("gazebo_msgs.srv")
    gazebo_msgs_srv.SetModelState = type("SetModelState", (), {})
    nav_msgs_msg = types.ModuleType("nav_msgs.msg")
    nav_msgs_msg.Odometry = type("Odometry", (), {})
    std_msgs_msg = types.ModuleType("std_msgs.msg")
    std_msgs_msg.Bool = type("Bool", (), {})
    for pkg, msg in (
        ("geometry_msgs", geometry_msgs_msg),
        ("gazebo_msgs", gazebo_msgs_msg),
        ("nav_msgs", nav_msgs_msg),
        ("std_msgs", std_msgs_msg),
    ):
        sys.modules[pkg] = types.ModuleType(pkg)
        sys.modules[pkg + ".msg"] = msg
    sys.modules["gazebo_msgs.srv"] = gazebo_msgs_srv


def _load_module():
    _install_ros_stubs()
    script = Path(__file__).resolve().parents[1] / "scripts" / "body_cmd_vel_driver.py"
    spec = importlib.util.spec_from_file_location("body_cmd_vel_driver_under_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BodyCmdVelDriverTest(unittest.TestCase):
    def test_planner_velocity_is_used_outside_scan_mode(self):
        mod = _load_module()
        self.assertEqual(
            mod.select_velocity((0.8, 0.1, 0.2), False, (0.0, 0.0, 1.0)),
            (0.8, 0.1, 0.2),
        )

    def test_scan_velocity_overrides_planner_in_scan_mode(self):
        mod = _load_module()
        self.assertEqual(
            mod.select_velocity((0.8, 0.1, 0.2), True, (0.0, 0.0, 1.0)),
            (0.0, 0.0, 1.0),
        )

    def test_linear_scale_applies_only_to_planner_translation(self):
        mod = _load_module()
        self.assertEqual(mod.apply_linear_scale((0.8, 0.2, 0.3), False, 1.25), (1.0, 0.25, 0.3))
        self.assertEqual(mod.apply_linear_scale((0.0, 0.0, 2.0), True, 1.25), (0.0, 0.0, 2.0))

    def test_planar_velocity_is_norm_limited_to_one_point_five(self):
        mod = _load_module()
        vx, vy = mod.limit_planar_velocity(1.5, 1.5, 1.5)
        self.assertAlmostEqual((vx * vx + vy * vy) ** 0.5, 1.5)
        self.assertEqual(mod.limit_planar_velocity(0.8, 0.2, 1.5), (0.8, 0.2))

    def test_integration_dt_uses_sim_time_instead_of_assumed_loop_rate(self):
        mod = _load_module()
        self.assertAlmostEqual(mod.integration_dt(10.08, 10.0, 30.0), 0.08)
        self.assertAlmostEqual(mod.integration_dt(10.5, 10.0, 30.0), 0.2)
        self.assertAlmostEqual(mod.integration_dt(10.0, None, 30.0), 1.0 / 30.0)

    def test_odom_only_initializes_driver_pose_once(self):
        mod = _load_module()
        self.assertTrue(mod.should_initialize_pose(False))
        self.assertFalse(mod.should_initialize_pose(True))


if __name__ == "__main__":
    unittest.main()
