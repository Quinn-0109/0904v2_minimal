#!/usr/bin/env python3
"""Level Mid-360 cloud + IMU for FAST-LIO without editing official sim files.

Reads raw /scan (PointCloud) and /livox/imu, applies optional mount-pitch
undo, and publishes FAST-LIO-compatible XYZI/time/ring clouds + IMU.

Default unpitch is 0: Gazebo Mid-360 + aggressive motion is more stable when
TARE consumes planar-stabilized /state_estimation. Set ~unpitch_rad:=-0.785
to experiment with gravity-aligned body frame.
"""

import math

import numpy as np
import rospy
from sensor_msgs.msg import Imu, PointCloud, PointCloud2, PointField


FIELDS_XYZITR = [
    PointField("x", 0, PointField.FLOAT32, 1),
    PointField("y", 4, PointField.FLOAT32, 1),
    PointField("z", 8, PointField.FLOAT32, 1),
    PointField("intensity", 12, PointField.FLOAT32, 1),
    # FAST-LIO's standard PointCloud2 path consumes seconds when
    # preprocess/timestamp_unit=SEC. The simulator publishes a complete scan
    # at one ROS stamp, so distribute points monotonically over its 10 Hz
    # acquisition interval for IMU motion compensation.
    PointField("time", 16, PointField.FLOAT32, 1),
    PointField("ring", 20, PointField.UINT16, 1),
]

XYZITR_DTYPE = np.dtype({
    "names": ["x", "y", "z", "intensity", "time", "ring"],
    "formats": ["<f4", "<f4", "<f4", "<f4", "<f4", "<u2"],
    "offsets": [0, 4, 8, 12, 16, 20],
    "itemsize": 22,
})


def _create_xyzitr_cloud(header, kept, scan_period, scan_lines,
                         model_acquisition_time):
    """Pack a FAST-LIO cloud without per-point Python tuple/struct work."""
    count = kept.shape[0]
    records = np.empty(count, dtype=XYZITR_DTYPE)
    records["x"] = kept[:, 0]
    records["y"] = kept[:, 1]
    records["z"] = kept[:, 2]
    records["intensity"].fill(0.0)

    if model_acquisition_time:
        # Keep the original float64 multiply-then-divide order before the
        # final FLOAT32 field conversion performed by create_cloud/struct.
        point_times = np.arange(count, dtype=np.float64)
        point_times *= float(scan_period)
        point_times /= max(count - 1, 1)
        records["time"] = point_times
    else:
        # Positive epsilon tells FAST-LIO not to infer a rotating Velodyne
        # sweep from azimuth. It is effectively zero.
        records["time"].fill(1.0e-6)

    rings = records["ring"]
    rings[:] = np.arange(count, dtype=np.uint16)
    if scan_lines <= count:
        np.remainder(rings, np.uint16(scan_lines), out=rings)

    return PointCloud2(
        header=header,
        height=1,
        width=count,
        fields=FIELDS_XYZITR,
        is_bigendian=False,
        point_step=XYZITR_DTYPE.itemsize,
        row_step=XYZITR_DTYPE.itemsize * count,
        data=records.tobytes(order="C"),
        is_dense=False,
    )


def rot_y(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


class FastLioSensorAdapter:
    def __init__(self):
        self._unpitch = float(rospy.get_param("~unpitch_rad", 0.0))
        self._r = rot_y(self._unpitch)
        # The IMU callback runs at 1 kHz. Scalar coefficients avoid NumPy
        # vector and matrix allocation on every sample.
        self._imu_cos = math.cos(self._unpitch)
        self._imu_sin = math.sin(self._unpitch)
        self._rotate_imu = bool(rospy.get_param("~rotate_imu", abs(self._unpitch) > 1e-6))
        self._blind = float(rospy.get_param("~blind", 0.35))
        self._max_range = float(rospy.get_param("~max_range", 40.0))
        self._min_elev = math.radians(float(rospy.get_param("~min_elev_deg", -70.0)))
        self._max_elev = math.radians(float(rospy.get_param("~max_elev_deg", 70.0)))
        self._scan_period = float(rospy.get_param("~scan_period_s", 0.1))
        self._scan_lines = max(1, int(rospy.get_param("~scan_lines", 6)))
        # SimEnv's Gazebo plugin updates every ray at one simulation instant;
        # point order represents the scan pattern, not acquisition over 0.1 s.
        # Assigning a fabricated sweep time causes metre-scale drift on spins.
        self._model_acquisition_time = bool(
            rospy.get_param("~model_acquisition_time", False)
        )
        self._raw_imu_topic = rospy.get_param("~raw_imu_topic", "/livox/imu")
        self._warmup_seconds = max(
            0.0, float(rospy.get_param("~warmup_seconds", 0.0))
        )
        self._first_sensor_stamp = None
        self._warmup_logged = False
        self._dropped_imu = 0
        self._sanitized_imu = 0
        self._max_acceleration = float(rospy.get_param("~max_acceleration_mps2", 45.0))
        self._max_angular_rate = float(rospy.get_param("~max_angular_rate_rps", 12.0))
        self._acc_filter_alpha = float(rospy.get_param("~acc_filter_alpha", 0.10))
        self._gyr_filter_alpha = float(rospy.get_param("~gyr_filter_alpha", 0.20))
        self._filtered_acc = None
        self._filtered_gyr = None

        cloud_topic = rospy.get_param("~cloud_topic", "/simenv/fast_lio/points")
        imu_topic = rospy.get_param("~imu_topic", "/simenv/fast_lio/imu")
        self._cloud_pub = rospy.Publisher(cloud_topic, PointCloud2, queue_size=2)
        self._imu_pub = rospy.Publisher(imu_topic, Imu, queue_size=50)

        rospy.Subscriber("/scan", PointCloud, self._on_scan, queue_size=2)
        rospy.Subscriber(self._raw_imu_topic, Imu, self._on_imu, queue_size=100)
        rospy.loginfo(
            "FAST-LIO sensor adapter: /scan+%s -> %s + %s "
            "(unpitch=%.3f rotate_imu=%s acquisition_time=%s)",
            self._raw_imu_topic,
            cloud_topic,
            imu_topic,
            self._unpitch,
            self._rotate_imu,
            "modeled" if self._model_acquisition_time else "instantaneous",
        )

    def _warmup_complete(self, stamp):
        seconds = stamp.to_sec()
        if seconds <= 0.0:
            return self._warmup_seconds <= 0.0
        if self._first_sensor_stamp is None:
            self._first_sensor_stamp = seconds
        ready = seconds - self._first_sensor_stamp >= self._warmup_seconds
        if ready and not self._warmup_logged:
            self._warmup_logged = True
            rospy.loginfo(
                "[STARTUP] FAST-LIO sensor warmup complete after %.1fs; "
                "starting IMU/LiDAR feed",
                self._warmup_seconds,
            )
        return ready

    def _on_scan(self, message):
        if rospy.is_shutdown():
            return
        if not message.points or not self._warmup_complete(message.header.stamp):
            return
        pts = np.array([[p.x, p.y, p.z] for p in message.points], dtype=np.float64)
        if abs(self._unpitch) > 1e-9:
            leveled = (self._r @ pts.T).T
        else:
            leveled = pts
        ranges = np.linalg.norm(leveled, axis=1)
        planar = np.linalg.norm(leveled[:, :2], axis=1)
        elev = np.arctan2(leveled[:, 2], np.maximum(planar, 1e-6))
        mask = (
            (ranges >= self._blind)
            & (ranges <= self._max_range)
            & (elev >= self._min_elev)
            & (elev <= self._max_elev)
            & np.isfinite(ranges)
        )
        kept = leveled[mask]
        if kept.size == 0:
            return
        if kept.shape[0] > 6000:
            idx = np.linspace(0, kept.shape[0] - 1, 6000).astype(np.int32)
            kept = kept[idx]
        header = message.header
        header.frame_id = "laser_livox_leveled"
        try:
            self._cloud_pub.publish(
                _create_xyzitr_cloud(
                    header,
                    kept,
                    self._scan_period,
                    self._scan_lines,
                    self._model_acquisition_time,
                )
            )
        except rospy.exceptions.ROSException:
            # roslaunch may close publishers while a callback is in flight.
            return

    def _on_imu(self, message):
        if rospy.is_shutdown():
            return
        if not self._warmup_complete(message.header.stamp):
            return
        out = Imu()
        out.header = message.header
        out.header.frame_id = "livox_imu"
        ax = float(message.linear_acceleration.x)
        ay = float(message.linear_acceleration.y)
        az = float(message.linear_acceleration.z)
        gx = float(message.angular_velocity.x)
        gy = float(message.angular_velocity.y)
        gz = float(message.angular_velocity.z)
        if self._rotate_imu:
            c, s = self._imu_cos, self._imu_sin
            ax, az = c * ax + s * az, -s * ax + c * az
            gx, gz = c * gx + s * gz, -s * gx + c * gz
        values = (ax, ay, az, gx, gy, gz)
        finite = all(math.isfinite(value) for value in values)
        acc_norm_sq = ax * ax + ay * ay + az * az
        gyr_norm_sq = gx * gx + gy * gy + gz * gz
        spike = (
            not finite
            or acc_norm_sq > self._max_acceleration ** 2
            or gyr_norm_sq > self._max_angular_rate ** 2
        )
        if spike and (self._filtered_acc is None or self._filtered_gyr is None):
            self._dropped_imu += 1
            rospy.logwarn_throttle(
                2.0, "Dropping invalid IMU spike (count=%d, a=%.2f, w=%.2f)",
                self._dropped_imu, math.sqrt(max(0.0, acc_norm_sq)),
                math.sqrt(max(0.0, gyr_norm_sq)),
            )
            return
        if spike:
            # Preserve the 1 kHz integration timeline with the previous
            # filtered state instead of dropping an impact sample.
            self._sanitized_imu += 1
            acc = list(self._filtered_acc)
            gyr = list(self._filtered_gyr)
            rospy.logwarn_throttle(
                2.0, "Sanitizing IMU impact (count=%d)", self._sanitized_imu,
            )
        else:
            acc = [ax, ay, az]
            gyr = [gx, gy, gz]
            if self._filtered_acc is None:
                self._filtered_acc = list(acc)
                self._filtered_gyr = list(gyr)
            else:
                aa, ga = self._acc_filter_alpha, self._gyr_filter_alpha
                for index in range(3):
                    self._filtered_acc[index] += (
                        aa * (acc[index] - self._filtered_acc[index]))
                    self._filtered_gyr[index] += (
                        ga * (gyr[index] - self._filtered_gyr[index]))
            acc = list(self._filtered_acc)
            gyr = list(self._filtered_gyr)
        out.linear_acceleration.x = float(acc[0])
        out.linear_acceleration.y = float(acc[1])
        out.linear_acceleration.z = float(acc[2])
        out.angular_velocity.x = float(gyr[0])
        out.angular_velocity.y = float(gyr[1])
        out.angular_velocity.z = float(gyr[2])
        out.orientation_covariance = message.orientation_covariance
        out.angular_velocity_covariance = message.angular_velocity_covariance
        out.linear_acceleration_covariance = message.linear_acceleration_covariance
        out.orientation.w = 1.0
        if not rospy.is_shutdown():
            try:
                self._imu_pub.publish(out)
            except rospy.exceptions.ROSException:
                # A callback can already be in flight while roslaunch closes
                # publishers during orderly shutdown.
                return


if __name__ == "__main__":
    rospy.init_node("fastlio_pointcloud_adapter")
    FastLioSensorAdapter()
    rospy.spin()
