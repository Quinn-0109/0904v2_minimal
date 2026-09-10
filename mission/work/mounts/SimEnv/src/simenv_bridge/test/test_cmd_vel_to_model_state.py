#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import sys
import types
import unittest


def _install_ros_stubs():
    sys.modules["rospy"] = types.ModuleType("rospy")
    gazebo_msgs_msg = types.ModuleType("gazebo_msgs.msg")
    gazebo_msgs_srv = types.ModuleType("gazebo_msgs.srv")
    geometry_msgs_msg = types.ModuleType("geometry_msgs.msg")

    class ModelState:
        def __init__(self):
            self.model_name = ""
            self.reference_frame = ""
            self.pose = None
            self.twist = types.SimpleNamespace(
                linear=types.SimpleNamespace(x=0.0, y=0.0, z=0.0),
                angular=types.SimpleNamespace(x=0.0, y=0.0, z=0.0),
            )

    gazebo_msgs_msg.ModelState = ModelState
    gazebo_msgs_srv.GetModelState = type("GetModelState", (), {})
    gazebo_msgs_srv.SetModelState = type("SetModelState", (), {})
    geometry_msgs_msg.Twist = type("Twist", (), {})
    sys.modules["gazebo_msgs"] = types.ModuleType("gazebo_msgs")
    sys.modules["gazebo_msgs.msg"] = gazebo_msgs_msg
    sys.modules["gazebo_msgs.srv"] = gazebo_msgs_srv
    sys.modules["geometry_msgs"] = types.ModuleType("geometry_msgs")
    sys.modules["geometry_msgs.msg"] = geometry_msgs_msg

    tf = types.ModuleType("tf")
    tf_trans = types.ModuleType("tf.transformations")

    def quaternion_from_euler(_roll, _pitch, yaw):
        import math
        return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))

    def euler_from_quaternion(q):
        import math
        _x, _y, z, w = q
        return (0.0, 0.0, 2.0 * math.atan2(z, w))

    tf_trans.quaternion_from_euler = quaternion_from_euler
    tf_trans.euler_from_quaternion = euler_from_quaternion
    sys.modules["tf"] = tf
    sys.modules["tf.transformations"] = tf_trans


def _load_module():
    _install_ros_stubs()
    script = Path(__file__).resolve().parents[1] / "scripts" / "cmd_vel_to_model_state.py"
    spec = importlib.util.spec_from_file_location("cmd_vel_to_model_state_under_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CmdVelToModelStateTest(unittest.TestCase):
    def test_make_next_state_integrates_world_velocity_and_levels_pose(self):
        mod = _load_module()
        pose = types.SimpleNamespace(
            position=types.SimpleNamespace(x=1.0, y=2.0, z=0.1),
            orientation=types.SimpleNamespace(x=0.2, y=-0.1, z=0.0, w=1.0),
        )

        state = mod.make_next_state(
            model_name="a1_gazebo",
            pose=pose,
            vx=-0.2,
            vy=0.1,
            wz=0.0,
            dt=2.0,
            command_frame="world",
            min_z=0.35,
        )

        self.assertEqual(state.model_name, "a1_gazebo")
        self.assertEqual(state.reference_frame, "world")
        self.assertAlmostEqual(state.pose.position.x, 0.6)
        self.assertAlmostEqual(state.pose.position.y, 2.2)
        self.assertAlmostEqual(state.pose.position.z, 0.35)
        self.assertEqual(state.pose.orientation.x, 0.0)
        self.assertEqual(state.pose.orientation.y, 0.0)
        self.assertAlmostEqual(state.twist.linear.x, -0.2)
        self.assertAlmostEqual(state.twist.linear.y, 0.1)

    def test_make_next_state_can_lock_height_to_target_z(self):
        mod = _load_module()
        pose = types.SimpleNamespace(
            position=types.SimpleNamespace(x=0.0, y=0.0, z=2.4),
            orientation=types.SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        )

        state = mod.make_next_state(
            model_name="a1_gazebo",
            pose=pose,
            vx=0.0,
            vy=0.0,
            wz=0.0,
            dt=0.1,
            command_frame="world",
            min_z=0.35,
            lock_z=0.6,
        )

        self.assertAlmostEqual(state.pose.position.z, 0.6)


if __name__ == "__main__":
    unittest.main()
