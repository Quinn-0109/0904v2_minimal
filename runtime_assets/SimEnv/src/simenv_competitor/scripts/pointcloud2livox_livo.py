#!/usr/bin/env python3
"""Convert leveled PointCloud2 to Livox CustomMsg for FAST-LIVO2."""

import struct

import rospy
from sensor_msgs.msg import PointCloud2, PointField
from unitree_guide.msg import CustomMsg, CustomPoint


def _struct_fmt(cloud):
    fmt = ""
    for field in cloud.fields:
        if field.datatype == PointField.FLOAT32:
            fmt += "f"
        elif field.datatype == PointField.UINT32:
            fmt += "I"
        elif field.datatype == PointField.UINT8:
            fmt += "B"
        elif field.datatype == PointField.UINT16:
            fmt += "H"
    return fmt


class PointCloud2LivoxAdapter:
    def __init__(self):
        in_topic = rospy.get_param("~input_topic", "/simenv/fast_lio/points")
        out_topic = rospy.get_param("~output_topic", "/simenv/livox/lidar")
        # Mid-360 scan period ~100 ms; distribute offsets for IMU undistortion.
        self._scan_period_ns = int(
            float(rospy.get_param("~scan_period_s", 0.1)) * 1e9
        )
        self._pub = rospy.Publisher(out_topic, CustomMsg, queue_size=2)
        rospy.Subscriber(in_topic, PointCloud2, self._on_cloud, queue_size=2)
        rospy.loginfo("LIVO2 lidar adapter: %s -> %s", in_topic, out_topic)

    def _on_cloud(self, cloud):
        if cloud.width == 0:
            return
        fmt = _struct_fmt(cloud)
        if len(fmt) < 3:
            return
        msg = CustomMsg()
        msg.header = cloud.header
        msg.timebase = cloud.header.stamp.to_nsec()
        msg.lidar_id = 1
        msg.rsvd = [0, 0, 0]
        points = []
        step = cloud.point_step
        n_pts = max(1, cloud.width * max(1, cloud.height))
        for idx, offset in enumerate(range(0, len(cloud.data), step)):
            if offset + 12 > len(cloud.data):
                break
            x, y, z = struct.unpack_from("<fff", cloud.data, offset)
            pt = CustomPoint()
            # Spread timestamps across one scan for motion compensation.
            pt.offset_time = int(idx * self._scan_period_ns / n_pts)
            pt.x = float(x)
            pt.y = float(y)
            pt.z = float(z)
            pt.reflectivity = 0
            # Non-feature Avia path accepts any line < N_SCANS; tag unused.
            pt.tag = 0x10
            pt.line = idx % 6
            points.append(pt)
        msg.points = points
        msg.point_num = len(points)
        self._pub.publish(msg)


if __name__ == "__main__":
    rospy.init_node("pointcloud2livox_livo")
    PointCloud2LivoxAdapter()
    rospy.spin()
