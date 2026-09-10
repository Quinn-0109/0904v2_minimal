#!/usr/bin/env python3
"""Bridge RealSense depth points into the TARE topic contract.

Used when Livox Gazebo plugin is unstable. Still an oracle baseline when
fed by /Odometry_gazebo.
"""

import copy

import rospy
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2


class RealsenseTareBridge:
    def __init__(self):
        self._latest_odom = None
        self._max_range = float(rospy.get_param("~max_range", 6.0))
        self._voxel = float(rospy.get_param("~voxel", 0.08))
        self._state_pub = rospy.Publisher("/state_estimation", Odometry, queue_size=5)
        self._scan_state_pub = rospy.Publisher(
            "/state_estimation_at_scan", Odometry, queue_size=5
        )
        self._scan_pub = rospy.Publisher("/registered_scan", PointCloud2, queue_size=2)
        self._cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=5)
        rospy.Subscriber("/Odometry_gazebo", Odometry, self._on_odom, queue_size=10)
        rospy.Subscriber(
            "/real_sense/depth/points", PointCloud2, self._on_cloud, queue_size=1
        )
        rospy.Subscriber(
            "/tare/cmd_vel_stamped", TwistStamped, self._on_cmd, queue_size=5
        )

    @staticmethod
    def _tare_odom(message, stamp=None):
        output = copy.deepcopy(message)
        output.header.frame_id = "map"
        output.child_frame_id = "sensor"
        if stamp is not None:
            output.header.stamp = stamp
        return output

    def _on_odom(self, message):
        self._latest_odom = message
        self._state_pub.publish(self._tare_odom(message))

    def _on_cloud(self, message):
        if self._latest_odom is None:
            return
        pose = self._latest_odom.pose.pose
        q = pose.orientation
        # Quaternion to rotation matrix.
        x, y, z, w = q.x, q.y, q.z, q.w
        r00 = 1 - 2 * (y * y + z * z)
        r01 = 2 * (x * y - z * w)
        r02 = 2 * (x * z + y * w)
        r10 = 2 * (x * y + z * w)
        r11 = 1 - 2 * (x * x + z * z)
        r12 = 2 * (y * z - x * w)
        r20 = 2 * (x * z - y * w)
        r21 = 2 * (y * z + x * w)
        r22 = 1 - 2 * (x * x + y * y)
        tx, ty, tz = pose.position.x, pose.position.y, pose.position.z

        # Camera optical: x right, y down, z forward.
        # A1 base/world: x forward, y left, z up.
        # Optical -> base: (z, -x, -y) then add camera mount.
        cam_x, cam_y, cam_z = 0.28, 0.0, 0.043
        points = []
        seen = set()
        inv_voxel = 1.0 / max(self._voxel, 1e-3)
        for px, py, pz in pc2.read_points(
            message, field_names=("x", "y", "z"), skip_nans=True
        ):
            if pz <= 0.05 or pz > self._max_range:
                continue
            bx = pz + cam_x
            by = -px + cam_y
            bz = -py + cam_z
            wx = r00 * bx + r01 * by + r02 * bz + tx
            wy = r10 * bx + r11 * by + r12 * bz + ty
            wz = r20 * bx + r21 * by + r22 * bz + tz
            key = (
                int(wx * inv_voxel),
                int(wy * inv_voxel),
                int(wz * inv_voxel),
            )
            if key in seen:
                continue
            seen.add(key)
            # local_planner PointXYZI expects intensity; use absolute z as a
            # stand-in height channel (unused when useTerrainAnalysis=false).
            points.append((wx, wy, wz, float(wz)))
            if len(points) >= 12000:
                break

        header = copy.deepcopy(message.header)
        header.frame_id = "map"
        fields = [
            pc2.PointField("x", 0, pc2.PointField.FLOAT32, 1),
            pc2.PointField("y", 4, pc2.PointField.FLOAT32, 1),
            pc2.PointField("z", 8, pc2.PointField.FLOAT32, 1),
            pc2.PointField("intensity", 12, pc2.PointField.FLOAT32, 1),
        ]
        cloud = pc2.create_cloud(header, fields, points)
        self._scan_pub.publish(cloud)
        self._scan_state_pub.publish(self._tare_odom(self._latest_odom, message.header.stamp))

    def _on_cmd(self, message):
        command = copy.deepcopy(message.twist)
        max_linear = rospy.get_param("~max_linear_speed", 0.8)
        max_yaw = rospy.get_param("~max_yaw_rate", 1.0)
        command.linear.x = max(-max_linear, min(max_linear, command.linear.x))
        command.linear.y = max(-max_linear, min(max_linear, command.linear.y))
        command.angular.z = max(-max_yaw, min(max_yaw, command.angular.z))
        self._cmd_pub.publish(command)


if __name__ == "__main__":
    rospy.init_node("realsense_tare_bridge")
    RealsenseTareBridge()
    rospy.spin()
