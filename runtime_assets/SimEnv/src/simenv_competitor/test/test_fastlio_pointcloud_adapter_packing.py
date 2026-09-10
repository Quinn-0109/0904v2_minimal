#!/usr/bin/env python3
"""Byte-for-byte contract for the vectorized FAST-LIO cloud packer."""

from io import BytesIO
import os
import sys
import unittest

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as point_cloud2
from std_msgs.msg import Header


SCRIPT_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "scripts"))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import fastlio_pointcloud_adapter as adapter


def _fixture(count):
    indices = np.arange(count, dtype=np.float64)
    kept = np.empty((count, 3), dtype=np.float64)
    kept[:, 0] = np.sin(indices * 0.173) * 19.75
    kept[:, 1] = np.cos(indices * 0.117) * -17.25
    kept[:, 2] = ((indices % 97.0) - 48.0) / 4.0
    kept[0] = (0.0, -0.0, 1.0)
    if count > 1:
        kept[-1] = (19.999999, -17.000001, 1.0e-6)
    return kept


def _header():
    return Header(
        seq=73,
        stamp=rospy.Time(secs=1234, nsecs=567890123),
        frame_id="laser_livox_leveled",
    )


def _reference_cloud(kept, scan_period, scan_lines,
                     model_acquisition_time):
    count = kept.shape[0]
    denominator = max(count - 1, 1)
    points = [
        (
            float(x),
            float(y),
            float(z),
            0.0,
            (
                float(index) * scan_period / denominator
                if model_acquisition_time
                else 1.0e-6
            ),
            int(index % scan_lines),
        )
        for index, (x, y, z) in enumerate(kept)
    ]
    return point_cloud2.create_cloud(
        _header(), adapter.FIELDS_XYZITR, points)


def _serialized(message):
    output = BytesIO()
    message.serialize(output)
    return output.getvalue()


def _metadata(message):
    return (
        message.header.seq,
        message.header.stamp.secs,
        message.header.stamp.nsecs,
        message.header.frame_id,
        message.height,
        message.width,
        tuple((field.name, field.offset, field.datatype, field.count)
              for field in message.fields),
        message.is_bigendian,
        message.point_step,
        message.row_step,
        message.is_dense,
    )


class FastLioPointCloudPackingTest(unittest.TestCase):
    def test_dtype_exactly_matches_ros_field_layout(self):
        self.assertEqual(adapter.XYZITR_DTYPE.itemsize, 22)
        expected = {
            "x": (np.dtype("<f4"), 0),
            "y": (np.dtype("<f4"), 4),
            "z": (np.dtype("<f4"), 8),
            "intensity": (np.dtype("<f4"), 12),
            "time": (np.dtype("<f4"), 16),
            "ring": (np.dtype("<u2"), 20),
        }
        self.assertEqual(adapter.XYZITR_DTYPE.fields, expected)

    def test_matches_create_cloud_bytes_metadata_and_serialization(self):
        for count in (1, 2, 799, 6000):
            kept = _fixture(count)
            ring_boundaries = (1, 6, count, count + 1, 65535)
            for modeled_time in (False, True):
                for scan_lines in sorted(set(ring_boundaries)):
                    with self.subTest(
                            count=count,
                            modeled_time=modeled_time,
                            scan_lines=scan_lines):
                        reference = _reference_cloud(
                            kept, 0.1, scan_lines, modeled_time)
                        actual = adapter._create_xyzitr_cloud(
                            _header(), kept, 0.1, scan_lines, modeled_time)

                        self.assertEqual(_metadata(actual),
                                         _metadata(reference))
                        self.assertEqual(actual.data, reference.data)
                        self.assertEqual(_serialized(actual),
                                         _serialized(reference))


if __name__ == "__main__":
    unittest.main()
