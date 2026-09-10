#!/usr/bin/env python3
"""Truth-registered cloud relay: raw /scan + truth odom -> /cloud_registered.

FAST-LIO normally registers each Mid-360 scan into its camera_init map frame
and publishes the result on /cloud_registered for fastlio_voxel_mapper.
Without FAST-LIO, this node performs the same geometric transform directly:

  p_camera_init = R_truth(t) * (R_base_laser * p_laser + t_base_laser)
                + t_truth(t)

where (R_truth, t_truth) is the latest ground-truth odometry pose (published
by ground_truth_odometry.py on /Odometry, world-frame values) and
base->laser_livox is looked up from the URDF TF published by
robot_state_publisher.  The output is stamped with the scan time so the
mapper can pair each cloud with the closest odometry sample, exactly as it
does for FAST-LIO's registered stream.

Filters mirror the FAST-LIO sensor adapter: blind distance and maximum range.
The mapper applies its own range/self filters afterwards.
"""

import math

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as point_cloud2
import tf
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud, PointCloud2, PointField

FIELDS_XYZ = [
    PointField("x", 0, PointField.FLOAT32, 1),
    PointField("y", 4, PointField.FLOAT32, 1),
    PointField("z", 8, PointField.FLOAT32, 1),
]


def _quat_to_matrix(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


class TruthRegisteredCloud:
    def __init__(self):
        self._scan_topic = str(rospy.get_param("~scan_topic", "/scan"))
        self._odom_topic = str(rospy.get_param(
            "~odom_topic", "/Odometry"))
        self._cloud_topic = str(rospy.get_param(
            "~cloud_topic", "/cloud_registered"))
        self._parent_frame = str(rospy.get_param("~parent_frame", "base"))
        self._sensor_frame = str(rospy.get_param(
            "~sensor_frame", "laser_livox"))
        self._output_frame = str(rospy.get_param(
            "~output_frame", "camera_init"))
        self._blind = float(rospy.get_param("~blind", 0.25))
        self._max_range = float(rospy.get_param("~max_range", 40.0))
        self._tf_timeout = float(rospy.get_param("~tf_timeout_sec", 5.0))
        self._max_odom_age = float(rospy.get_param(
            "~max_odom_age_sec", 0.25))

        self._odom = None  # (stamp_sec, pos, rot_matrix)
        self._mount_ready = False
        self._mount_translation = np.zeros(3, dtype=np.float64)
        self._mount_rotation = np.eye(3, dtype=np.float64)

        self._cloud_pub = rospy.Publisher(
            self._cloud_topic, PointCloud2, queue_size=5)
        rospy.Subscriber(self._scan_topic, PointCloud,
                         self._on_scan, queue_size=4)
        rospy.Subscriber(self._odom_topic, Odometry,
                         self._on_odom, queue_size=50)
        self._listener = tf.TransformListener()
        self._scan_count = 0
        self._dropped_no_mount = 0
        self._dropped_no_odom = 0
        rospy.loginfo("Truth registered cloud ready: %s + %s -> %s "
                      "(%s->%s, output %s)",
                      self._scan_topic, self._odom_topic, self._cloud_topic,
                      self._parent_frame, self._sensor_frame,
                      self._output_frame)

    def _ensure_mount(self):
        if self._mount_ready:
            return True
        try:
            stamp = rospy.Time(0)
            (trans, rot) = self._listener.lookupTransform(
                self._parent_frame, self._sensor_frame, stamp)
        except (tf.LookupException, tf.ConnectivityException,
                tf.ExtrapolationException):
            if self._dropped_no_mount < 20:
                rospy.logwarn_throttle(
                    5.0, "Truth cloud waiting for TF %s->%s",
                    self._parent_frame, self._sensor_frame)
            return False
        self._mount_translation = np.array(trans, dtype=np.float64)
        self._mount_rotation = _quat_to_matrix(rot[0], rot[1], rot[2], rot[3])
        self._mount_ready = True
        rospy.loginfo("Truth cloud mount base->%s: t=(%.4f, %.4f, %.4f)",
                      self._sensor_frame, *self._mount_translation)
        return True

    def _on_odom(self, message):
        p = message.pose.pose.position
        q = message.pose.pose.orientation
        if not all(math.isfinite(v) for v in
                   (p.x, p.y, p.z, q.x, q.y, q.z, q.w)):
            return
        stamp_sec = message.header.stamp.to_sec()
        if stamp_sec <= 0.0:
            stamp_sec = rospy.Time.now().to_sec()
        self._odom = (stamp_sec,
                      np.array([p.x, p.y, p.z], dtype=np.float64),
                      _quat_to_matrix(q.x, q.y, q.z, q.w))

    def _on_scan(self, message):
        if rospy.is_shutdown():
            return
        if not self._ensure_mount():
            self._dropped_no_mount += 1
            return
        if self._odom is None:
            self._dropped_no_odom += 1
            return
        scan_stamp = message.header.stamp.to_sec()
        if scan_stamp <= 0.0:
            scan_stamp = rospy.Time.now().to_sec()
        odom_stamp, t_truth, r_truth = self._odom
        if abs(scan_stamp - odom_stamp) > self._max_odom_age:
            self._dropped_no_odom += 1
            return
        if not message.points:
            return
        pts = np.array([[p.x, p.y, p.z] for p in message.points],
                       dtype=np.float64)
        if pts.size == 0:
            return
        # Body frame, then truth pose into the navigation frame.
        body = (self._mount_rotation @ pts.T).T + self._mount_translation
        nav = (r_truth @ body.T).T + t_truth
        ranges = np.linalg.norm(body, axis=1)
        mask = ((ranges >= self._blind) & (ranges <= self._max_range)
                & np.isfinite(ranges))
        kept = nav[mask]
        if kept.size == 0:
            return
        header = message.header
        header.frame_id = self._output_frame
        try:
            self._cloud_pub.publish(point_cloud2.create_cloud(
                header, FIELDS_XYZ, kept.tolist()))
        except rospy.exceptions.ROSException:
            return
        self._scan_count += 1
        if self._scan_count <= 5:
            rospy.loginfo("Truth cloud #%d published: %d points "
                          "(stamp %.3f, odom %.3f)",
                          self._scan_count, kept.shape[0], scan_stamp,
                          odom_stamp)


if __name__ == "__main__":
    rospy.init_node("truth_registered_cloud")
    TruthRegisteredCloud()
    rospy.spin()
