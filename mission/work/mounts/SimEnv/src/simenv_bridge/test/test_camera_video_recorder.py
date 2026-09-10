#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest

import cv2
import numpy as np


def _install_ros_stubs():
    sys.modules["rospy"] = types.ModuleType("rospy")
    sensor_msgs_msg = types.ModuleType("sensor_msgs.msg")
    sensor_msgs_msg.Image = type("Image", (), {})
    std_msgs_msg = types.ModuleType("std_msgs.msg")
    std_msgs_msg.Bool = type("Bool", (), {})
    for pkg, msg in (("sensor_msgs", sensor_msgs_msg), ("std_msgs", std_msgs_msg)):
        sys.modules[pkg] = types.ModuleType(pkg)
        sys.modules[pkg + ".msg"] = msg


def _load_module(testcase):
    _install_ros_stubs()
    script = Path(__file__).resolve().parents[1] / "scripts" / "camera_video_recorder.py"
    testcase.assertTrue(script.exists(), "camera_video_recorder.py is missing")
    spec = importlib.util.spec_from_file_location("camera_video_recorder_under_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CameraVideoRecorderTest(unittest.TestCase):
    def test_converts_rgb_and_bgr_messages_to_bgr(self):
        mod = _load_module(self)
        rgb = types.SimpleNamespace(width=2, height=1, encoding="rgb8",
                                    data=bytes([255, 0, 0, 0, 255, 0]))
        bgr = types.SimpleNamespace(width=2, height=1, encoding="bgr8",
                                    data=bytes([0, 0, 255, 0, 255, 0]))
        expected = np.array([[[0, 0, 255], [0, 255, 0]]], dtype=np.uint8)
        np.testing.assert_array_equal(mod.image_msg_to_bgr(rgb), expected)
        np.testing.assert_array_equal(mod.image_msg_to_bgr(bgr), expected)

    def test_rejects_unsupported_or_malformed_message(self):
        mod = _load_module(self)
        with self.assertRaises(ValueError):
            mod.image_msg_to_bgr(types.SimpleNamespace(width=1, height=1, encoding="mono8", data=b"\x00"))
        with self.assertRaises(ValueError):
            mod.image_msg_to_bgr(types.SimpleNamespace(width=2, height=1, encoding="rgb8", data=b"\x00"))

    def test_ten_fps_throttle(self):
        mod = _load_module(self)
        self.assertTrue(mod.should_write_frame(None, 0.00, 10.0))
        self.assertFalse(mod.should_write_frame(0.00, 0.05, 10.0))
        self.assertTrue(mod.should_write_frame(0.00, 0.10, 10.0))
        self.assertTrue(mod.should_write_frame(0.10, 0.21, 10.0))

    def test_writer_helper_produces_decodable_mp4(self):
        mod = _load_module(self)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "camera.mp4")
            writer = mod.open_video_writer(path, 64, 48, 10.0)
            for index in range(10):
                frame = np.full((48, 64, 3), index * 20, dtype=np.uint8)
                writer.write(frame)
            writer.release()
            capture = cv2.VideoCapture(path)
            self.assertTrue(capture.isOpened())
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 64)
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), 48)
            self.assertGreaterEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 9)
            capture.release()


if __name__ == "__main__":
    unittest.main()
