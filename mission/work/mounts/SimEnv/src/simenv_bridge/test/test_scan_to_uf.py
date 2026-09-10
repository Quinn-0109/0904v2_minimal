#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import sys
import types
import unittest


class PointField:
    FLOAT32 = 7
    FLOAT64 = 8
    UINT8 = 2

    def __init__(self, name="", offset=0, datatype=0, count=1):
        self.name = name
        self.offset = offset
        self.datatype = datatype
        self.count = count


class PointCloud2:
    pass


class Header:
    def __init__(self):
        self.stamp = types.SimpleNamespace(secs=12, nsecs=345)
        self.frame_id = "laser_livox"


def _install_ros_stubs(points):
    rospy = types.ModuleType("rospy")
    rospy.Time = lambda *_args, **_kwargs: None
    sys.modules["rospy"] = rospy

    std_msgs = types.ModuleType("std_msgs")
    std_msgs_msg = types.ModuleType("std_msgs.msg")
    std_msgs_msg.Header = Header
    std_msgs.msg = std_msgs_msg
    sys.modules["std_msgs"] = std_msgs
    sys.modules["std_msgs.msg"] = std_msgs_msg

    sensor_msgs = types.ModuleType("sensor_msgs")
    sensor_msgs_msg = types.ModuleType("sensor_msgs.msg")
    sensor_msgs_pc2 = types.ModuleType("sensor_msgs.point_cloud2")
    sensor_msgs_msg.PointCloud2 = PointCloud2
    sensor_msgs_msg.PointField = PointField
    sensor_msgs_pc2.read_points = lambda *_args, **_kwargs: iter(points)
    sensor_msgs_pc2.create_cloud = lambda header, fields, pts: types.SimpleNamespace(
        header=header,
        fields=fields,
        points=pts,
    )
    sys.modules["sensor_msgs"] = sensor_msgs
    sys.modules["sensor_msgs.msg"] = sensor_msgs_msg
    sys.modules["sensor_msgs.point_cloud2"] = sensor_msgs_pc2

    unitree_guide = types.ModuleType("unitree_guide")
    unitree_guide_msg = types.ModuleType("unitree_guide.msg")

    class CustomPoint:
        def __init__(self):
            self.x = 0.0
            self.y = 0.0
            self.z = 0.0
            self.reflectivity = 0
            self.offset_time = 0
            self.tag = 0
            self.line = 0

    class CustomMsg:
        def __init__(self):
            self.header = Header()
            self.points = []
            self.timebase = 0
            self.point_num = 0
            self.lidar_id = 0
            self.rsvd = [0, 0, 0]

    unitree_guide_msg.CustomMsg = CustomMsg
    unitree_guide_msg.CustomPoint = CustomPoint
    sys.modules["unitree_guide"] = unitree_guide
    sys.modules["unitree_guide.msg"] = unitree_guide_msg


def _load_module(points):
    _install_ros_stubs(points)
    script = Path(__file__).resolve().parents[1] / "scripts" / "scan_to_uf.py"
    spec = importlib.util.spec_from_file_location("scan_to_uf_under_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ScanToUfTest(unittest.TestCase):
    def test_pointcloud_output_matches_ultra_fusion_livox_pointcloud_layout(self):
        mod = _load_module([(1.0, 2.0, 0.5, 7.0), (2.0, -1.0, -0.25, 3.0)])
        cloud = PointCloud2()
        cloud.header = Header()
        cloud.fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
            PointField("intensity", 12, PointField.FLOAT32, 1),
        ]

        out = mod.adapt_cloud(cloud, scan_duration_ns=20, total_lines=16, min_range=0.0)

        self.assertEqual([field.name for field in out.fields], ["x", "y", "z", "intensity", "timestamp", "tag", "line"])
        self.assertEqual(out.point_step, 26)
        self.assertEqual(out.row_step, 52)

    def test_custom_output_matches_ultra_fusion_livox_pointcloud_layout(self):
        mod = _load_module([])
        custom = sys.modules["unitree_guide.msg"].CustomMsg()
        point = sys.modules["unitree_guide.msg"].CustomPoint()
        point.x = 1.0
        point.y = 0.0
        point.z = 0.2
        point.reflectivity = 11
        point.offset_time = 5
        point.tag = 2
        point.line = 6
        custom.points.append(point)

        out = mod.adapt_custom_msg(custom, min_range=0.0)

        self.assertEqual([field.name for field in out.fields], ["x", "y", "z", "intensity", "timestamp", "tag", "line"])
        self.assertEqual(out.point_step, 26)
        self.assertEqual(out.row_step, 26)


if __name__ == "__main__":
    unittest.main()
