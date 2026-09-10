#!/usr/bin/env python3
"""
scan_to_uf.py - Publish an Ultra-Fusion friendly PointCloud2.

Default input is PointCloud2 because Ultra-Fusion's ROS2 bag conversion path
stores Livox packets as sensor_msgs/PointCloud2 with per-point timestamp/tag/line.
The node can still consume Livox-style CustomMsg when explicitly requested.
"""

import math
import struct
import sys

import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from unitree_guide.msg import CustomMsg

F32 = PointField.FLOAT32
F64 = PointField.FLOAT64
U8 = PointField.UINT8
POINT_STEP = 26
FIELDS = [
    PointField(name="x", offset=0, datatype=F32, count=1),
    PointField(name="y", offset=4, datatype=F32, count=1),
    PointField(name="z", offset=8, datatype=F32, count=1),
    PointField(name="intensity", offset=12, datatype=F32, count=1),
    PointField(name="timestamp", offset=16, datatype=F64, count=1),
    PointField(name="tag", offset=24, datatype=U8, count=1),
    PointField(name="line", offset=25, datatype=U8, count=1),
]


def _get_field_names(msg):
    return [field.name for field in msg.fields]


def _field_index(msg, name):
    try:
        return _get_field_names(msg).index(name)
    except ValueError:
        return -1


def _iter_points(msg):
    names = _get_field_names(msg)
    if "intensity" in names:
        field_names = ("x", "y", "z", "intensity")
    else:
        field_names = ("x", "y", "z")

    for point in pc2.read_points(msg, field_names=field_names, skip_nans=True):
        if len(point) == 4:
            yield point
        else:
            x, y, z = point
            yield x, y, z, 1.0


def _iter_custom_points(msg, min_range):
    min_range_sq = min_range * min_range
    for point in msg.points:
        x = float(point.x)
        y = float(point.y)
        z = float(point.z)
        if (x * x + y * y + z * z) < min_range_sq:
            continue
        yield x, y, z, float(point.reflectivity), float(point.offset_time), int(point.tag) & 0xFF, int(point.line) & 0xFF


def _synthesize_line(x, y, z, total_lines):
    horiz = math.hypot(x, y)
    pitch = math.degrees(math.atan2(z, horiz if horiz > 1e-6 else 1e-6))
    normalized = max(0.0, min(1.0, (pitch + 30.0) / 90.0))
    line = int(round(normalized * (total_lines - 1)))
    return max(0, min(total_lines - 1, line))


def adapt_cloud(msg, stamp_base_ns=None, scan_duration_ns=100_000_000, total_lines=32, min_range=0.2):
    points = []
    min_range_sq = min_range * min_range
    for x, y, z, intensity in _iter_points(msg):
        if (x * x + y * y + z * z) < min_range_sq:
            continue
        points.append((x, y, z, intensity))
    out = PointCloud2()
    out.header = msg.header
    out.height = 1
    out.width = len(points)
    out.fields = FIELDS
    out.is_bigendian = False
    out.point_step = POINT_STEP
    out.row_step = POINT_STEP * len(points)
    out.is_dense = True

    if stamp_base_ns is None:
        stamp = msg.header.stamp
        stamp_base_ns = int(stamp.secs) * 1_000_000_000 + int(stamp.nsecs)

    buf = bytearray(out.row_step)
    count = max(1, len(points))
    for index, (x, y, z, intensity) in enumerate(points):
        offset = index * POINT_STEP
        rel_ns = int(index * scan_duration_ns / count)
        # Ultra-Fusion's Livox PointCloud2 path expects per-point offset time
        # in nanoseconds, not an absolute ROS timestamp.
        timestamp_ns = float(rel_ns)
        line = _synthesize_line(x, y, z, total_lines)
        struct.pack_into(
            "<ffffdBB",
            buf,
            offset,
            float(x),
            float(y),
            float(z),
            float(intensity),
            timestamp_ns,
            0,
            line & 0xFF,
        )

    out.data = bytes(buf)
    return out


def adapt_custom_msg(msg, min_range=0.2):
    points = list(_iter_custom_points(msg, min_range))
    out = PointCloud2()
    out.header = msg.header
    out.height = 1
    out.width = len(points)
    out.fields = FIELDS
    out.is_bigendian = False
    out.point_step = POINT_STEP
    out.row_step = POINT_STEP * len(points)
    out.is_dense = True

    buf = bytearray(out.row_step)
    for index, (x, y, z, intensity, offset_time, tag, line) in enumerate(points):
        struct.pack_into(
            "<ffffdBB",
            buf,
            index * POINT_STEP,
            x,
            y,
            z,
            intensity,
            offset_time,
            tag,
            line,
        )
    out.data = bytes(buf)
    return out


def should_publish_fallback(now_sec, last_custom_sec, custom_timeout_sec):
    if last_custom_sec is None:
        return True
    return (now_sec - last_custom_sec) > custom_timeout_sec


def _self_test():
    header = Header()
    header.stamp = rospy.Time(12, 345)
    header.frame_id = "laser_livox"
    cloud = pc2.create_cloud(
        header,
        [
            PointField("x", 0, F32, 1),
            PointField("y", 4, F32, 1),
            PointField("z", 8, F32, 1),
            PointField("intensity", 12, F32, 1),
        ],
        [
            (1.0, 2.0, 0.5, 7.0),
            (2.0, -1.0, -0.25, 3.0),
        ],
    )
    out = adapt_cloud(cloud, stamp_base_ns=12_000_000_345, scan_duration_ns=20, total_lines=16, min_range=0.0)

    assert out.width == 2
    assert out.point_step == POINT_STEP
    assert [field.name for field in out.fields] == [
        "x",
        "y",
        "z",
        "intensity",
        "timestamp",
        "tag",
        "line",
    ]

    first = struct.unpack_from("<ffffdBB", out.data, 0)
    second = struct.unpack_from("<ffffdBB", out.data, POINT_STEP)
    assert first[:4] == (1.0, 2.0, 0.5, 7.0)
    assert second[:4] == (2.0, -1.0, -0.25, 3.0)
    assert first[4] == 0.0
    assert second[4] == 10.0
    assert first[5] == 0 and second[5] == 0
    assert 0 <= first[6] < 16
    assert 0 <= second[6] < 16
    print("scan_to_uf self-test passed")

    custom = CustomMsg()
    custom.header = header
    custom.timebase = 0
    custom.point_num = 2
    custom.lidar_id = 1
    custom.rsvd = [0, 0, 0]
    for x, y, z, reflectivity, offset_time, line in [
        (1.0, 0.0, 0.2, 11, 0, 5),
        (2.0, 0.0, 0.3, 22, 10, 6),
    ]:
        from unitree_guide.msg import CustomPoint
        p = CustomPoint()
        p.x = x
        p.y = y
        p.z = z
        p.reflectivity = reflectivity
        p.offset_time = offset_time
        p.tag = 0
        p.line = line
        custom.points.append(p)
    out2 = adapt_custom_msg(custom, min_range=0.0)
    first2 = struct.unpack_from("<ffffdBB", out2.data, 0)
    second2 = struct.unpack_from("<ffffdBB", out2.data, POINT_STEP)
    assert all(abs(a - b) < 1e-6 for a, b in zip(first2[:4], (1.0, 0.0, 0.2, 11.0)))
    assert all(abs(a - b) < 1e-6 for a, b in zip(second2[:4], (2.0, 0.0, 0.3, 22.0)))
    assert first2[4] == 0.0 and second2[4] == 10.0
    assert first2[5] == 0 and second2[5] == 0
    assert first2[6] == 5 and second2[6] == 6
    assert should_publish_fallback(10.0, None, 1.0)
    assert not should_publish_fallback(10.0, 9.5, 1.0)
    assert should_publish_fallback(10.0, 8.9, 1.0)


def main():
    if "--self-test" in sys.argv:
        _self_test()
        return

    rospy.init_node("scan_to_uf")
    inp = rospy.get_param("~input", "/scan")
    out = rospy.get_param("~output", "/scan_uf")
    custom_input = rospy.get_param("~custom_input", "/livox/lidar2")
    scan_duration = int(rospy.get_param("~scan_duration_ns", 100_000_000))
    total_lines = int(rospy.get_param("~total_rings", rospy.get_param("~total_lines", 32)))
    min_range = float(rospy.get_param("~min_range", 0.2))
    custom_timeout = float(rospy.get_param("~custom_timeout", 1.0))
    pub = rospy.Publisher(out, PointCloud2, queue_size=10)
    counter = {"n": 0, "custom": 0, "fallback": 0}
    last_custom = {"t": None}

    def cb(msg):
        if rospy.get_param("~prefer_custom", True):
            now_sec = rospy.Time.now().to_sec()
            if not should_publish_fallback(now_sec, last_custom["t"], custom_timeout):
                return
        adapted = adapt_cloud(
            msg,
            scan_duration_ns=scan_duration,
            total_lines=total_lines,
            min_range=min_range,
        )
        pub.publish(adapted)
        counter["n"] += 1
        counter["fallback"] += 1
        if counter["n"] % 50 == 0:
            rospy.loginfo("scan_to_uf: %d clouds republished to %s (custom=%d fallback=%d)",
                          counter["n"], out, counter["custom"], counter["fallback"])

    def cb_custom(msg):
        last_custom["t"] = rospy.Time.now().to_sec()
        adapted = adapt_custom_msg(msg, min_range=min_range)
        pub.publish(adapted)
        counter["n"] += 1
        counter["custom"] += 1
        if counter["n"] % 50 == 0:
            rospy.loginfo("scan_to_uf: %d custom clouds republished to %s (custom=%d fallback=%d)",
                          counter["n"], out, counter["custom"], counter["fallback"])

    if rospy.get_param("~prefer_custom", True):
        rospy.Subscriber(custom_input, CustomMsg, cb_custom, queue_size=10)
        rospy.Subscriber(inp, PointCloud2, cb, queue_size=10)
        rospy.loginfo("scan_to_uf: %s -> %s via CustomMsg with %s fallback after %.1fs (point_step=%d, min_range=%.2f)",
                      custom_input, out, inp, custom_timeout, POINT_STEP, min_range)
    else:
        rospy.Subscriber(inp, PointCloud2, cb, queue_size=10)
        rospy.loginfo("scan_to_uf: %s -> %s via PointCloud2 (point_step=%d, min_range=%.2f)", inp, out, POINT_STEP, min_range)
    rospy.spin()


if __name__ == "__main__":
    main()
