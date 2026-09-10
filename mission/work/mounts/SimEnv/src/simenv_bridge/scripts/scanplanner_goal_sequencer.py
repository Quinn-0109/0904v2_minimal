#!/usr/bin/env python3
"""Door-aligned four-room mission driver for SCAN-Planner navi_mode=1."""
import json
import math
import time

import rospy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, String


def _goal(label, x, y):
    return {"kind": "goal", "label": label, "x": float(x), "y": float(y)}


def _scan(label, turn=2.0 * math.pi):
    return {"kind": "scan", "label": label, "angular_z": 2.0, "turn": float(turn)}


def _fallback_layout():
    rooms = []
    for index, (side, door_y) in enumerate(
        (("left", 14.865), ("right", 14.865), ("left", 28.895), ("right", 28.895))
    ):
        sign = -1.0 if side == "left" else 1.0
        rooms.append({
            "id": "floor_0_room_%d" % index,
            "side": side,
            "door_pose": [sign * 1.1, door_y],
            "goal_pose": [sign * 4.8, door_y],
            "bounds": {
                "x_min": -9.5 if side == "left" else 1.1,
                "x_max": -1.1 if side == "left" else 9.5,
                "y_min": door_y - 7.015,
                "y_max": door_y + 7.015,
            },
        })
    return {"floors": [{"floor_index": 0, "rooms": rooms}]}


def load_layout(path):
    with open(path, "r") as stream:
        return json.load(stream)


def build_route(layout=None):
    """Build an unbiased two-station sweep for every ground-floor room."""
    layout = layout or _fallback_layout()
    floor = min(layout["floors"], key=lambda value: int(value.get("floor_index", 0)))
    rooms = sorted(
        floor.get("rooms", []),
        key=lambda room: (float(room["door_pose"][1]), 0 if room.get("side") == "left" else 1),
    )
    route = []
    for room in rooms:
        prefix = str(room["id"])
        side = str(room.get("side", "left"))
        sign = -1.0 if side == "left" else 1.0
        door_x, door_y = map(float, room["door_pose"][:2])
        goal_x = float(room["goal_pose"][0])
        bounds = room["bounds"]
        low_y = max(float(bounds["y_min"]) + 2.0, door_y - 5.0)
        high_y = min(float(bounds["y_max"]) - 2.0, door_y + 5.0)
        entry_x = door_x + sign * 0.9

        route.extend([
            _goal(prefix + "_corridor", 0.0, door_y),
            _goal(prefix + "_entry", entry_x, door_y),
            _goal(prefix + "_center", goal_x, door_y),
            _scan(prefix + "_scan_center"),
            _goal(prefix + "_low_center", goal_x, low_y),
            _scan(prefix + "_scan_low"),
            _goal(prefix + "_high_center", goal_x, high_y),
            _scan(prefix + "_scan_high"),
            _goal(prefix + "_return_center", goal_x, door_y),
            _goal(prefix + "_return_entry", entry_x, door_y),
            _goal(prefix + "_return_corridor", 0.0, door_y),
        ])
    return route


def accumulated_rotation(previous_yaw, current_yaw):
    return abs(math.atan2(math.sin(current_yaw - previous_yaw), math.cos(current_yaw - previous_yaw)))


def _yaw_from_odom(msg):
    q = msg.pose.pose.orientation
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def goal_tolerance(action, default_tolerance):
    """Use a larger acceptance radius for room-center waypoints.

    A room center is only a scan staging point; accepting it at 1.5 m keeps
    the route moving while still requiring corridor and door waypoints to be
    reached within the tighter default tolerance.
    """
    label = str(action.get("label", ""))
    if label.startswith("room_") and label.endswith("_center"):
        return max(float(default_tolerance), 1.5)
    return float(default_tolerance)


class MissionSequencer:
    def __init__(self):
        self.goal_tol = float(rospy.get_param("~goal_tol", 1.0))
        self.rate_hz = float(rospy.get_param("~rate", 3.0))
        self.stall_seconds = float(rospy.get_param("~stall_seconds", 20.0))
        layout_path = rospy.get_param(
            "~layout_path", "/workspace/SimEnv/generated_building/layout_metadata.json"
        )
        try:
            self.route = build_route(load_layout(layout_path))
        except Exception as exc:
            rospy.logwarn("sequencer: failed to load %s (%s); using four-room fallback", layout_path, exc)
            self.route = build_route()
        rospy.loginfo("sequencer: generated unbiased route with %d actions", len(self.route))
        self.position = None
        self.yaw = 0.0
        self.goal_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1, latch=True)
        self.scan_active_pub = rospy.Publisher("/scanplanner/scan_active", Bool, queue_size=1, latch=True)
        self.scan_cmd_pub = rospy.Publisher("/scanplanner/scan_cmd_vel", Twist, queue_size=1)
        self.state_pub = rospy.Publisher("/scanplanner/route_state", String, queue_size=1, latch=True)
        self.complete_pub = rospy.Publisher("/scanplanner/route_complete", Bool, queue_size=1, latch=True)
        self.failed_pub = rospy.Publisher("/scanplanner/route_failed", Bool, queue_size=1, latch=True)
        rospy.Subscriber("/Odometry_gazebo", Odometry, self._odom_cb, queue_size=50)

    def _odom_cb(self, msg):
        p = msg.pose.pose.position
        self.position = (p.x, p.y)
        self.yaw = _yaw_from_odom(msg)

    def _publish_goal(self, action):
        msg = PoseStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = rospy.Time.now()
        msg.pose.position.x = action["x"]
        msg.pose.position.y = action["y"]
        msg.pose.orientation.w = 1.0
        self.goal_pub.publish(msg)

    def _publish_scan(self, active, angular_z=0.0):
        self.scan_active_pub.publish(Bool(data=active))
        cmd = Twist()
        cmd.angular.z = angular_z
        self.scan_cmd_pub.publish(cmd)

    def _previous_corridor(self, index):
        for action in reversed(self.route[:index]):
            if action["kind"] == "goal" and action["label"].startswith("corridor"):
                return dict(action)
        return None

    def run(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown() and self.position is None:
            rate.sleep()

        index = 0
        best_distance = float("inf")
        last_progress = time.monotonic()
        scan_total = 0.0
        scan_previous_yaw = self.yaw
        scan_started = False
        recovery_counts = {}
        self.complete_pub.publish(Bool(data=False))
        self.failed_pub.publish(Bool(data=False))

        while not rospy.is_shutdown() and index < len(self.route):
            action = self.route[index]
            self.state_pub.publish(String(data=action["label"]))

            if action["kind"] == "scan":
                if not scan_started:
                    scan_started = True
                    scan_total = 0.0
                    scan_previous_yaw = self.yaw
                    rospy.loginfo("sequencer: scan %s", action["label"])
                scan_total += accumulated_rotation(scan_previous_yaw, self.yaw)
                scan_previous_yaw = self.yaw
                self._publish_scan(True, action["angular_z"])
                if scan_total >= action["turn"]:
                    self._publish_scan(False)
                    rospy.loginfo("sequencer: completed %s turn=%.2f", action["label"], scan_total)
                    index += 1
                    scan_started = False
                    best_distance = float("inf")
                    last_progress = time.monotonic()
                rate.sleep()
                continue

            self._publish_scan(False)
            self._publish_goal(action)
            distance = math.hypot(self.position[0] - action["x"], self.position[1] - action["y"])
            if distance < best_distance - 0.4:
                best_distance = distance
                last_progress = time.monotonic()
            if distance < goal_tolerance(action, self.goal_tol):
                rospy.loginfo("sequencer: reached %s d=%.2f", action["label"], distance)
                index += 1
                best_distance = float("inf")
                last_progress = time.monotonic()
                rate.sleep()
                continue
            if time.monotonic() - last_progress > self.stall_seconds:
                label = action["label"]
                if recovery_counts.get(label, 0) == 0:
                    recovery = self._previous_corridor(index)
                    if recovery is not None:
                        recovery["label"] = "recovery_for_" + label
                        self.route.insert(index, recovery)
                        recovery_counts[label] = 1
                        best_distance = float("inf")
                        last_progress = time.monotonic()
                        rospy.logwarn("sequencer: stalled at %s; recover via corridor", label)
                    else:
                        recovery_counts[label] = 1
                else:
                    self._publish_scan(False)
                    self.failed_pub.publish(Bool(data=True))
                    self.state_pub.publish(String(data="failed:" + label))
                    rospy.logerr("sequencer: repeated stall at %s", label)
                    rospy.spin()
                    return
            rate.sleep()

        self._publish_scan(False)
        self.complete_pub.publish(Bool(data=True))
        self.state_pub.publish(String(data="completed"))
        rospy.loginfo("sequencer: mission completed")
        rospy.spin()


def main():
    rospy.init_node("scanplanner_goal_sequencer", anonymous=False)
    MissionSequencer().run()


if __name__ == "__main__":
    main()
