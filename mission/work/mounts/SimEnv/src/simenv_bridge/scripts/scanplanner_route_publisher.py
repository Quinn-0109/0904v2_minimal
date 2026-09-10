#!/usr/bin/env python3
"""Publish boustrophedon waypoints as nav_msgs/Path on /initial_path for SCAN-Planner
navi_mode=3 (reference-path tracking + local obstacle avoidance).

Reads layout_metadata.json -> entrance -> lobby -> corridor -> for each room:
corridor@door-y -> through-door -> room-goal -> back-to-corridor -> corridor end.
One connected reference path; SCAN-Planner fits a single global B-spline through it
(one-shot, no chatter) -> reaches all rooms. This is the verified-working approach.
"""
import json
import sys
import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped


def main():
    meta = sys.argv[1] if len(sys.argv) > 1 else '/workspace/SimEnv/generated_building/layout_metadata.json'
    m = json.load(open(meta))
    f0 = m['floors'][0]
    pts = []
    lb = f0['lobby_bounds']
    pts.append((0.0, max(1.0, lb['y_min'] + 1.0)))           # entrance
    pts.append((0.0, (lb['y_min'] + lb['y_max']) / 2))       # lobby center
    cb = f0['corridor_bounds']
    pts.append((0.0, cb['y_min'] + 1.0))                      # corridor start
    for r in f0.get('rooms', []):
        dp = r['door_pose'][:2]
        gp = r['goal_pose'][:2]
        pts.append((0.0, dp[1]))       # corridor at door y
        pts.append((dp[0], dp[1]))     # through door
        pts.append((gp[0], gp[1]))     # room goal
        pts.append((0.0, dp[1]))       # back to corridor
    pts.append((0.0, cb['y_max'] - 1.0))                      # corridor end

    rospy.init_node('route_publisher', anonymous=True)
    pub = rospy.Publisher('/initial_path', Path, queue_size=1, latch=True)
    path = Path()
    path.header.frame_id = 'map'
    path.header.stamp = rospy.Time.now()
    for x, y in pts:
        ps = PoseStamped()
        ps.header = path.header
        ps.pose.position.x = float(x)
        ps.pose.position.y = float(y)
        ps.pose.position.z = 0.0
        ps.pose.orientation.w = 1.0
        path.poses.append(ps)
    rospy.sleep(1.0)
    pub.publish(path)
    rospy.loginfo("route_publisher: %d waypoints on /initial_path" % len(path.poses))
    rospy.spin()


if __name__ == '__main__':
    main()
