#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import sys
import types
import unittest


def _install_ros_stubs():
    sys.modules["rospy"] = types.ModuleType("rospy")
    nav_msgs_msg = types.ModuleType("nav_msgs.msg")
    nav_msgs_msg.Odometry = type("Odometry", (), {})
    sys.modules["nav_msgs"] = types.ModuleType("nav_msgs")
    sys.modules["nav_msgs.msg"] = nav_msgs_msg


def _load_module():
    _install_ros_stubs()
    script = Path(__file__).resolve().parents[1] / "scripts" / "odom_relay.py"
    spec = importlib.util.spec_from_file_location("odom_relay_under_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OdomRelayTest(unittest.TestCase):
    def test_relay_odom_sets_frame_and_optional_locked_z(self):
        mod = _load_module()
        msg = types.SimpleNamespace(
            header=types.SimpleNamespace(frame_id="odom"),
            pose=types.SimpleNamespace(
                pose=types.SimpleNamespace(position=types.SimpleNamespace(x=1.0, y=2.0, z=2.4))
            ),
        )

        out = mod.prepare_odom(msg, frame="map", lock_z=0.6)

        self.assertIs(out, msg)
        self.assertEqual(out.header.frame_id, "map")
        self.assertAlmostEqual(out.pose.pose.position.z, 0.6)

    def test_relay_odom_preserves_z_when_lock_is_disabled(self):
        mod = _load_module()
        msg = types.SimpleNamespace(
            header=types.SimpleNamespace(frame_id="odom"),
            pose=types.SimpleNamespace(
                pose=types.SimpleNamespace(position=types.SimpleNamespace(x=1.0, y=2.0, z=2.4))
            ),
        )

        out = mod.prepare_odom(msg, frame="map", lock_z=None)

        self.assertAlmostEqual(out.pose.pose.position.z, 2.4)


if __name__ == "__main__":
    unittest.main()
