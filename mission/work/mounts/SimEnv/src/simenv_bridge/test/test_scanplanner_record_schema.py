#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import sys
import types
import unittest

import numpy as np


def _install_ros_stubs():
    rospy = types.ModuleType("rospy")
    sys.modules["rospy"] = rospy
    nav_msgs_msg = types.ModuleType("nav_msgs.msg")
    nav_msgs_msg.Odometry = type("Odometry", (), {})
    sensor_msgs_msg = types.ModuleType("sensor_msgs.msg")
    sensor_msgs_msg.PointCloud2 = type("PointCloud2", (), {})
    sensor_pc2 = types.ModuleType("sensor_msgs.point_cloud2")
    sensor_pc2.read_points = lambda *args, **kwargs: []
    geometry_msgs_msg = types.ModuleType("geometry_msgs.msg")
    geometry_msgs_msg.PoseArray = type("PoseArray", (), {})
    geometry_msgs_msg.Twist = type("Twist", (), {})
    std_msgs_msg = types.ModuleType("std_msgs.msg")
    std_msgs_msg.Bool = type("Bool", (), {})
    std_msgs_msg.String = type("String", (), {})
    for pkg, msg in (
        ("nav_msgs", nav_msgs_msg),
        ("sensor_msgs", sensor_msgs_msg),
        ("geometry_msgs", geometry_msgs_msg),
        ("std_msgs", std_msgs_msg),
    ):
        sys.modules[pkg] = types.ModuleType(pkg)
        sys.modules[pkg + ".msg"] = msg
    sys.modules["sensor_msgs.point_cloud2"] = sensor_pc2


def _load_module():
    _install_ros_stubs()
    script = Path(__file__).resolve().parents[1] / "scripts" / "scanplanner_record.py"
    spec = importlib.util.spec_from_file_location("scanplanner_record_under_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ScanplannerRecordSchemaTest(unittest.TestCase):
    def test_pose_array_is_converted_to_xy_snapshot(self):
        mod = _load_module()
        msg = types.SimpleNamespace(poses=[
            types.SimpleNamespace(position=types.SimpleNamespace(x=1.0, y=2.0)),
            types.SimpleNamespace(position=types.SimpleNamespace(x=3.0, y=4.0)),
        ])
        np.testing.assert_allclose(mod.pose_array_xy(msg), [[1.0, 2.0], [3.0, 4.0]])

    def test_payload_contains_detection_and_route_fields(self):
        mod = _load_module()
        payload = mod.build_record_payload(
            trajectory=[(1.0, 0.0, 2.0, 1.57)],
            map_points=[(0.0, 0.0)],
            scan_samples=[],
            detection_samples=[(1.0, np.array([[3.0, 4.0]]))],
            state_samples=[(1.0, "room_LF_scan")],
            final_status="completed",
            applied_commands=[(1.0, 1.5)],
        )
        for key in ("det_t", "detections", "state_t", "states", "final_status", "cmd_t", "cmd_speed"):
            self.assertIn(key, payload)
        self.assertEqual(payload["final_status"].item(), "completed")
        self.assertEqual(payload["cmd_speed"].tolist(), [1.5])

    def test_truth_evaluation_renderer_declares_detection_overlay(self):
        renderer = Path(__file__).resolve().parents[1] / "scripts" / "plot_red_ball_truth_evaluation.py"
        text = renderer.read_text()
        self.assertIn('detection_record.get("detections"', text)
        self.assertIn("false_positive", text)
        self.assertIn("true_positive", text)


if __name__ == "__main__":
    unittest.main()
