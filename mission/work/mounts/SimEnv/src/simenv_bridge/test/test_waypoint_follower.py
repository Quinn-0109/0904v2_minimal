#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import sys
import types
import unittest


def _install_ros_stubs():
    sys.modules["rospy"] = types.ModuleType("rospy")
    sys.modules["tf2_ros"] = types.ModuleType("tf2_ros")
    sys.modules["tf2_geometry_msgs"] = types.ModuleType("tf2_geometry_msgs")
    geometry_msgs = types.ModuleType("geometry_msgs")
    geometry_msgs_msg = types.ModuleType("geometry_msgs.msg")
    geometry_msgs_msg.PointStamped = type("PointStamped", (), {})
    geometry_msgs_msg.Twist = type("Twist", (), {})
    sys.modules["geometry_msgs"] = geometry_msgs
    sys.modules["geometry_msgs.msg"] = geometry_msgs_msg


def _load_module():
    _install_ros_stubs()
    script = Path(__file__).resolve().parents[1] / "scripts" / "waypoint_follower.py"
    spec = importlib.util.spec_from_file_location("waypoint_follower_under_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WaypointFollowerTest(unittest.TestCase):
    def test_compute_cmd_from_base_point_drives_forward_for_ahead_target(self):
        mod = _load_module()

        vx, wz, done = mod.compute_cmd_from_base_point(2.0, 0.0, 0.8, 1.2, 0.5, 0.6, 0.3)

        self.assertFalse(done)
        self.assertGreater(vx, 0.0)
        self.assertAlmostEqual(wz, 0.0, places=6)

    def test_compute_cmd_from_base_point_turns_left_for_left_target(self):
        mod = _load_module()

        vx, wz, done = mod.compute_cmd_from_base_point(1.0, 1.0, 0.8, 1.2, 0.5, 0.6, 0.3)

        self.assertFalse(done)
        self.assertGreater(vx, 0.0)
        self.assertGreater(wz, 0.0)

    def test_compute_cmd_from_base_point_stops_near_goal(self):
        mod = _load_module()

        vx, wz, done = mod.compute_cmd_from_base_point(0.1, 0.1, 0.8, 1.2, 0.5, 0.6, 0.3)

        self.assertTrue(done)
        self.assertEqual(vx, 0.0)
        self.assertEqual(wz, 0.0)

    def test_tf_follower_default_speed_is_one_meter_per_second(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "waypoint_follower.py"
        text = script.read_text()

        self.assertIn('rospy.get_param("~v_max", 1.0)', text)


if __name__ == "__main__":
    unittest.main()
