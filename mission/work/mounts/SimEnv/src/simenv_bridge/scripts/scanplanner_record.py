#!/usr/bin/env python3
"""Record SCAN-Planner trajectory, lidar, mission state, and detections."""
import math
import time

import numpy as np
import rospy
from geometry_msgs.msg import PoseArray, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Bool, String


traj = []
explored = {}
explored_list = []
scans = []
detection_samples = []
state_samples = []
command_samples = []
last_scan_t = 0.0
terminal_status = None
SCAN_EVERY = 0.5
VOX = 0.12
GMAP_CAP = 40000


def yaw_from_q(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def pose_array_xy(msg):
    return np.asarray([(pose.position.x, pose.position.y) for pose in msg.poses], dtype=float).reshape(-1, 2)


def _object_array(values):
    output = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        output[index] = value
    return output


def build_record_payload(trajectory, map_points, scan_samples, detection_samples, state_samples,
                         final_status, applied_commands=None):
    applied_commands = applied_commands or []
    scan_xy = _object_array([sample[1] for sample in scan_samples])
    detections = _object_array([sample[1] for sample in detection_samples])
    return {
        "traj": np.asarray(trajectory, dtype=float).reshape(-1, 4),
        "gmap": np.asarray(map_points, dtype=float).reshape(-1, 2),
        "scan_t": np.asarray([sample[0] for sample in scan_samples], dtype=float),
        "scans": scan_xy,
        "det_t": np.asarray([sample[0] for sample in detection_samples], dtype=float),
        "detections": detections,
        "state_t": np.asarray([sample[0] for sample in state_samples], dtype=float),
        "states": np.asarray([sample[1] for sample in state_samples], dtype=object),
        "final_status": np.asarray(str(final_status)),
        "cmd_t": np.asarray([sample[0] for sample in applied_commands], dtype=float),
        "cmd_speed": np.asarray([sample[1] for sample in applied_commands], dtype=float),
    }


def on_odom(msg):
    p = msg.pose.pose.position
    traj.append((rospy.get_time(), p.x, p.y, yaw_from_q(msg.pose.pose.orientation)))


def _add(xy):
    if len(explored_list) >= GMAP_CAP:
        return
    key = (round(xy[0] / VOX), round(xy[1] / VOX))
    if key not in explored:
        explored[key] = xy
        explored_list.append(xy)


def on_scan(msg):
    global last_scan_t
    stamp = rospy.get_time()
    # Drop excess callbacks before decoding thousands of points.  The record
    # is evidence/visualisation, not a second online mapping pipeline.
    if stamp - last_scan_t < SCAN_EVERY:
        return
    last_scan_t = stamp
    points = np.asarray(list(pc2.read_points(msg, field_names=("x", "y"), skip_nans=True)))
    if points.size == 0:
        return
    xy = points[::3]
    for point in xy[::2]:
        _add(point)
    scans.append((stamp, xy))


def on_detections(msg):
    detection_samples.append((rospy.get_time(), pose_array_xy(msg)))


def on_state(msg):
    if not state_samples or state_samples[-1][1] != msg.data:
        state_samples.append((rospy.get_time(), msg.data))


def on_applied_cmd(msg):
    command_samples.append((rospy.get_time(), math.hypot(msg.linear.x, msg.linear.y)))


def on_complete(msg):
    global terminal_status
    if msg.data:
        terminal_status = "completed"


def on_failed(msg):
    global terminal_status
    if msg.data:
        terminal_status = "failed"


def main():
    global terminal_status, SCAN_EVERY
    rospy.init_node("scanplanner_record", anonymous=False)
    output = rospy.get_param("~out", "/tmp/scanplanner_record.npz")
    duration = float(rospy.get_param("~duration", 200.0))
    SCAN_EVERY = max(0.1, float(rospy.get_param(
        "~scan_interval_sec", SCAN_EVERY)))
    rospy.Subscriber("/Odometry_gazebo", Odometry, on_odom, queue_size=200)
    rospy.Subscriber("/registered_scan", PointCloud2, on_scan, queue_size=1, buff_size=1 << 24)
    rospy.Subscriber("/scanplanner/confirmed_dangers", PoseArray, on_detections, queue_size=10)
    rospy.Subscriber("/scanplanner/route_state", String, on_state, queue_size=10)
    rospy.Subscriber("/scanplanner/applied_cmd_vel", Twist, on_applied_cmd, queue_size=50)
    rospy.Subscriber("/scanplanner/route_complete", Bool, on_complete, queue_size=1)
    rospy.Subscriber("/scanplanner/route_failed", Bool, on_failed, queue_size=1)
    started = time.monotonic()
    rate = rospy.Rate(10)
    while not rospy.is_shutdown() and terminal_status is None and time.monotonic() - started < duration:
        rate.sleep()
    if terminal_status is None:
        terminal_status = "timed_out"
    payload = build_record_payload(
        traj, explored_list, scans, detection_samples, state_samples, terminal_status, command_samples
    )
    np.savez_compressed(output, **payload)
    rospy.loginfo("scanplanner_record: status=%s traj=%d map=%d scans=%d detections=%d -> %s",
                  terminal_status, len(traj), len(explored_list), len(scans), len(detection_samples), output)


if __name__ == "__main__":
    main()
