#!/usr/bin/env python3
"""ROS-light tests for dog-eye RGB video conversion and encoding."""

import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

import cv2
import numpy as np


SCRIPT_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "scripts"))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from rgb_video_recorder import RGBVideoRecorder


class RGBVideoRecorderTest(unittest.TestCase):
    def test_fixed_fps_slots_fill_slow_camera_callbacks(self):
        interval = 0.125
        self.assertEqual(
            RGBVideoRecorder.due_timeline_slots(None, 10.0, interval), 1)
        self.assertEqual(
            RGBVideoRecorder.due_timeline_slots(10.125, 10.10, interval), 0)
        self.assertEqual(
            RGBVideoRecorder.due_timeline_slots(10.125, 10.20, interval), 1)
        # Four 8-FPS slots are due after a half-second callback gap; the
        # recorder fills the first three from the preceding camera frame.
        self.assertEqual(
            RGBVideoRecorder.due_timeline_slots(10.125, 10.50, interval), 4)

    def test_rgb8_with_row_padding_decodes_to_bgr(self):
        # Two RGB pixels plus two padding bytes in every row.
        message = SimpleNamespace(
            encoding="rgb8", height=1, width=2, step=8,
            data=bytes([255, 0, 0, 0, 255, 0, 99, 99]))
        frame = RGBVideoRecorder.decode_image(message)
        self.assertEqual(frame.shape, (1, 2, 3))
        self.assertEqual(frame[0, 0].tolist(), [0, 0, 255])
        self.assertEqual(frame[0, 1].tolist(), [0, 255, 0])

    def test_mono8_decodes_to_three_channels(self):
        message = SimpleNamespace(
            encoding="mono8", height=1, width=2, step=2,
            data=bytes([12, 240]))
        frame = RGBVideoRecorder.decode_image(message)
        self.assertEqual(frame[0, 0].tolist(), [12, 12, 12])
        self.assertEqual(frame[0, 1].tolist(), [240, 240, 240])

    def test_mp4v_writer_is_available(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "probe.mp4")
            writer = cv2.VideoWriter(
                path, cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (64, 48))
            self.assertTrue(writer.isOpened())
            writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
            writer.release()
            self.assertGreater(os.path.getsize(path), 0)

    def test_launch_uses_independent_recording_camera_with_fallback(self):
        root = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
        with open(os.path.join(
                root, "simenv_competitor", "launch",
                "fuel_semantic_fastlio_exploration.launch")) as stream:
            launch = stream.read()
        with open(os.path.join(
                root, "unitree_guide", "unitree_ros", "robots",
                "a1_description", "xacro", "robot.xacro")) as stream:
            robot = stream.read()
        self.assertIn(
            '<param name="rgb_topic" value="/recording_camera/image_raw"/>',
            launch)
        self.assertIn(
            '<param name="fallback_rgb_topic" '
            'value="/real_sense/rgb/image_raw"/>', launch)
        self.assertIn('libgazebo_ros_camera.so', robot)
        self.assertIn('<cameraName>recording_camera</cameraName>', robot)


if __name__ == "__main__":
    unittest.main()
