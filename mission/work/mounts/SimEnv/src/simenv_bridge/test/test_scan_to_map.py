#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import sys
import types
import unittest


class PointField:
    FLOAT32 = 7

    def __init__(self, name="", offset=0, datatype=0, count=0):
        self.name = name
        self.offset = offset
        self.datatype = datatype
        self.count = count


class PointCloud2:
    pass


def _install_ros_stubs(points):
    sys.modules["rospy"] = types.ModuleType("rospy")
    sys.modules["tf2_ros"] = types.ModuleType("tf2_ros")
    sys.modules["tf2_sensor_msgs"] = types.ModuleType("tf2_sensor_msgs")
    sys.modules["tf2_sensor_msgs"].do_transform_cloud = lambda msg, _tf: msg
    sensor_msgs = types.ModuleType("sensor_msgs")
    sensor_msgs_msg = types.ModuleType("sensor_msgs.msg")
    sensor_msgs_pc2 = types.ModuleType("sensor_msgs.point_cloud2")
    sensor_msgs_msg.PointCloud2 = PointCloud2
    sensor_msgs_msg.PointField = PointField
    sensor_msgs_pc2.read_points = lambda *_args, **_kwargs: iter(points)
    sys.modules["sensor_msgs"] = sensor_msgs
    sys.modules["sensor_msgs.msg"] = sensor_msgs_msg
    sys.modules["sensor_msgs.point_cloud2"] = sensor_msgs_pc2


def _load_module(points):
    _install_ros_stubs(points)
    script = Path(__file__).resolve().parents[1] / "scripts" / "scan_to_map.py"
    spec = importlib.util.spec_from_file_location("scan_to_map_under_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ScanToMapTest(unittest.TestCase):
    def test_filter_range_discards_self_hits_and_far_points(self):
        points = [
            (0.1, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (7.0, 0.0, 0.0),
        ]
        mod = _load_module(points)
        cloud = PointCloud2()
        cloud.header = types.SimpleNamespace(frame_id="laser", stamp=None)
        cloud.fields = []

        filtered = mod._filter_range(cloud, min_range=0.5, max_range=6.0)

        self.assertEqual(filtered.width, 1)
        self.assertEqual(filtered.point_step, 16)
        self.assertEqual(len(filtered.data), 16)

    def test_filter_range_is_applied_before_map_transform(self):
        points = [
            (0.2, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (8.0, 0.0, 0.0),
        ]
        mod = _load_module(points)
        cloud = PointCloud2()
        cloud.header = types.SimpleNamespace(frame_id="laser", stamp=None)
        cloud.fields = []

        filtered = mod._prepare_sensor_range_cloud(cloud, min_range=0.5, max_range=6.0)

        self.assertEqual(filtered.width, 1)


if __name__ == "__main__":
    unittest.main()
