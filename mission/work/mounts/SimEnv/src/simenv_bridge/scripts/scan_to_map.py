#!/usr/bin/env python3
"""
scan_to_map.py — 为 TARE 生成 map 系的 registered_scan。

TARE 需要 /registered_scan (sensor_msgs/PointCloud2)，且点云必须已经在世界系(map)下；
TARE 自身不做配准/坐标变换。

本节点订阅 SimEnv 的点云 (默认 /real_sense/depth/points, frame=real_sense)，
用 tf2 把它从传感器系变换到 map 系，并补 intensity 字段，发布到 /registered_scan。

关键: SimEnv 已发布完整 TF 链 map -> odom -> base -> real_sense / laser_livox,
      所以用 tf2_sensor_msgs.do_transform_cloud 做真正的刚体变换。
      (不能只改 frame_id —— 那样点云位置会错。)

用法: rosrun simenv_bridge scan_to_map.py
        _input:=/real_sense/depth/points  _output:=/registered_scan  _target_frame:=map
"""
import struct
import rospy
import tf2_ros
import sensor_msgs.point_cloud2 as pc2
from tf2_sensor_msgs import do_transform_cloud
from sensor_msgs.msg import PointCloud2, PointField

PCL_F32 = 7  # PointField FLOAT32


def _filter_range(cloud, min_range, max_range):
    """剔除过近自身命中和过远点；TARE 只应基于局部感知推进探索。"""
    pts = list(pc2.read_points(cloud, field_names=("x", "y", "z"), skip_nans=True))
    min_sq = min_range * min_range
    max_sq = max_range * max_range if max_range > 0 else None
    keep = []
    for p in pts:
        dist_sq = p[0] * p[0] + p[1] * p[1] + p[2] * p[2]
        if dist_sq < min_sq:
            continue
        if max_sq is not None and dist_sq > max_sq:
            continue
        keep.append(p)
    fields = [
        PointField(name="x", offset=0, datatype=PCL_F32, count=1),
        PointField(name="y", offset=4, datatype=PCL_F32, count=1),
        PointField(name="z", offset=8, datatype=PCL_F32, count=1),
        PointField(name="intensity", offset=12, datatype=PCL_F32, count=1),
    ]
    out = PointCloud2()
    out.header = cloud.header
    out.height = 1
    out.width = len(keep)
    out.fields = fields
    out.is_bigendian = False
    out.point_step = 16
    out.row_step = 16 * len(keep)
    out.is_dense = True
    import struct
    out.data = b"".join(struct.pack("ffff", x, y, z, 1.0) for x, y, z in keep)
    return out


def _prepare_sensor_range_cloud(cloud, min_range, max_range):
    """在传感器原始坐标系中做距离裁剪，避免用 map 坐标误裁剪。"""
    return _filter_range(_ensure_intensity(cloud), min_range, max_range)


def _ensure_intensity(cloud):
    """若点云缺 intensity 字段，重建为 (x,y,z,intensity)。"""
    if any(f.name == "intensity" for f in cloud.fields):
        return cloud
    pts = list(pc2.read_points(cloud, field_names=("x", "y", "z"), skip_nans=True))
    fields = [
        PointField(name="x", offset=0, datatype=PCL_F32, count=1),
        PointField(name="y", offset=4, datatype=PCL_F32, count=1),
        PointField(name="z", offset=8, datatype=PCL_F32, count=1),
        PointField(name="intensity", offset=12, datatype=PCL_F32, count=1),
    ]
    out = PointCloud2()
    out.header = cloud.header
    out.height = 1
    out.width = len(pts)
    out.fields = fields
    out.is_bigendian = False
    out.point_step = 16
    out.row_step = 16 * len(pts)
    out.is_dense = True
    buf = bytearray()
    for x, y, z in pts:
        buf += struct.pack("ffff", x, y, z, 1.0)
    out.data = bytes(buf)
    return out


def main():
    rospy.init_node("scan_to_map")
    inp = rospy.get_param("~input", "/real_sense/depth/points")
    out = rospy.get_param("~output", "/registered_scan")
    target = rospy.get_param("~target_frame", "map")

    tfbuf = tf2_ros.Buffer()
    tfl = tf2_ros.TransformListener(tfbuf)
    pub = rospy.Publisher(out, PointCloud2, queue_size=10)
    min_range = float(rospy.get_param("~min_range", 0.5))  # 过滤近距命中(机器狗自身躯干/腿)
    max_range = float(rospy.get_param("~max_range", 6.0))  # TARE 局部规划范围；<=0 表示不过滤远距
    # pointcloud2livox can already project every return into odom/map with
    # Gazebo odometry.  Applying a sensor-origin range filter to those world
    # coordinates incorrectly removes the whole cloud once the robot is more
    # than max_range metres from (0, 0).  In that mode the converter has
    # already applied its own blind-range filter, so only add intensity here.
    input_is_world = bool(rospy.get_param("~input_is_world", False))
    count = {"n": 0}

    def cb(msg):
        src_frame = msg.header.frame_id or target
        if input_is_world:
            # The ground-truth pointcloud2livox path has already evaluated
            # P_world = R_world_body * P_body + t_world_body.  It labels that
            # numeric world frame "odom", but mapping_test intentionally does
            # not publish a map->odom TF (its only alias is map->camera_init).
            # Looking up a transform here therefore drops every cloud.  Trust
            # the explicitly selected world-coordinate contract and relabel it
            # as the planner's map frame without applying a second transform.
            msg = _ensure_intensity(msg)
            msg.header.frame_id = target
        else:
            msg = _prepare_sensor_range_cloud(msg, min_range, max_range)
        if not input_is_world and src_frame != target:
            try:
                tf = tfbuf.lookup_transform(target, src_frame, msg.header.stamp if msg.header.stamp else rospy.Time(0), rospy.Duration(0.1))
                msg = do_transform_cloud(msg, tf)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as e:
                return  # TF 暂时不可用, 跳过这一帧
        msg.header.frame_id = target
        pub.publish(msg)
        count["n"] += 1
        if count["n"] % 50 == 0:
            rospy.loginfo("scan_to_map: %d clouds (%d pts, %.1fm<range<%.1fm) -> %s",
                          count["n"], msg.width, min_range, max_range, out)

    rospy.Subscriber(inp, PointCloud2, cb, queue_size=10)
    rospy.loginfo("scan_to_map: %s -> %s (tf to %s, +intensity, range_filter=%s)",
                  inp, out, target, "upstream" if input_is_world else
                  "%.1fm..%.1fm" % (min_range, max_range))
    rospy.spin()


if __name__ == "__main__":
    main()
