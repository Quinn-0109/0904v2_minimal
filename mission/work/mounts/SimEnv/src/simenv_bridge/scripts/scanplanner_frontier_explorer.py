#!/usr/bin/env python3
"""Frontier-based exploration driver for SCAN-Planner (navi_mode=1) in SimEnv.

Reads SCAN-Planner's own /grid_map/unknown (unknown voxels of its sliding grid map),
clusters them into frontier regions, and publishes the nearest reachable frontier
centroid as /move_base_simple/goal. Because the goal is always at the known/unknown
boundary (~within the local planner's horizon) rather than a fixed far goal, the
planner can commit to it -- this is what gets the robot past the corridor stalls
that defeat a fixed room sequence. On stall (frontier behind a wall) it blacklists
the region and picks the next; stops when no frontiers remain.

Stays alive the whole run (SCAN-Planner stalls if the goal publisher exits).
Pure numpy/python clustering (no sklearn/scipy dep).
"""
import math
import time
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

VOX = 0.8          # clustering cell size (m)
REACH_MIN = 1.5    # ignore frontier cells closer than this (trivially near robot)
REACH_MAX = 12.0
STALL_T = 20.0
STALL_D = 0.4
BLACKLIST_R = 1.5
SEED = (0.0, 12.0)
SEED_T = 25.0


class FrontierExplorer:
    def __init__(self):
        rospy.init_node("frontier_explorer", anonymous=True)
        self.goal_tol = rospy.get_param("~goal_tol", 1.2)
        self.pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1, latch=True)
        self.robot = None
        self.unknown = np.zeros((0, 2))
        self.blacklist = []
        self.last_move_t = time.time()
        self.last_pos = None
        self.current_goal = None
        self.t0 = time.time()
        rospy.Subscriber("/Odometry_gazebo", Odometry, self.odom_cb, queue_size=50)
        rospy.Subscriber("/grid_map/unknown", PointCloud2, self.unknown_cb, queue_size=1, buff_size=1 << 24)

    def odom_cb(self, m):
        p = m.pose.pose.position
        self.robot = (p.x, p.y)
        if self.last_pos is None:
            self.last_pos = self.robot
        if math.hypot(self.robot[0] - self.last_pos[0], self.robot[1] - self.last_pos[1]) > STALL_D:
            self.last_move_t = time.time()
            self.last_pos = self.robot

    def unknown_cb(self, m):
        pts = np.array(list(pc2.read_points(m, field_names=("x", "y"), skip_nans=True)))
        if len(pts):
            self.unknown = pts

    def clusters(self):
        if self.robot is None or len(self.unknown) == 0:
            return []
        rx, ry = self.robot
        d = np.hypot(self.unknown[:, 0] - rx, self.unknown[:, 1] - ry)
        pts = self.unknown[(d > REACH_MIN) & (d < REACH_MAX)]
        if len(pts) == 0:
            return []
        cells = {}
        for x, y in pts[::2]:
            cells.setdefault((round(x / VOX), round(y / VOX)), []).append((x, y))
        comps = []
        seen = set()
        for k in list(cells.keys()):
            if k in seen:
                continue
            q = [k]; seen.add(k); comp = []
            while q:
                c = q.pop(); comp.append(c)
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        nk = (c[0] + dx, c[1] + dy)
                        if nk in cells and nk not in seen:
                            seen.add(nk); q.append(nk)
            if len(comp) >= 2:
                xs = [pt[0] for c in comp for pt in cells[c]]
                ys = [pt[1] for c in comp for pt in cells[c]]
                comps.append((float(np.mean(xs)), float(np.mean(ys)), len(comp)))
        return comps

    def blacklisted(self, cx, cy):
        return any(math.hypot(cx - bx, cy - by) < BLACKLIST_R for bx, by in self.blacklist)

    def pick_goal(self):
        comps = self.clusters()
        rx, ry = self.robot
        cands = [(math.hypot(cx - rx, cy - ry), cx, cy, n) for cx, cy, n in comps if not self.blacklisted(cx, cy)]
        if not cands:
            return None
        cands.sort()
        return (cands[0][1], cands[0][2], cands[0][3])

    def run(self):
        rate = rospy.Rate(2.0)
        while not rospy.is_shutdown() and self.robot is None:
            rate.sleep()
        rospy.loginfo("frontier: start at (%.1f,%.1f)", *self.robot)
        while not rospy.is_shutdown():
            # stall -> blacklist current
            if self.current_goal and (time.time() - self.last_move_t) > STALL_T:
                rospy.loginfo("frontier: stall near (%.1f,%.1f) -> blacklist", *self.current_goal[:2])
                self.blacklist.append(self.current_goal[:2])
                self.current_goal = None
                self.last_move_t = time.time()
            # reached current -> clear
            if self.current_goal:
                dd = math.hypot(self.robot[0] - self.current_goal[0], self.robot[1] - self.current_goal[1])
                if dd < self.goal_tol:
                    rospy.loginfo("frontier: reached (%.1f,%.1f) -> next", *self.current_goal[:2])
                    self.current_goal = None
            # pick next if needed
            if self.current_goal is None:
                g = self.pick_goal()
                if g:
                    self.current_goal = g
                    rospy.loginfo("frontier: -> (%.1f,%.1f) n=%d", *g)
                elif time.time() - self.t0 < SEED_T:
                    self.current_goal = (SEED[0], SEED[1], 0)  # bootstrap nudge
                    rospy.loginfo("frontier: seed -> (%.1f,%.1f)", *SEED)
                else:
                    rospy.loginfo_throttle(5.0, "frontier: no candidates (explored / all blacklisted)")
            # publish
            if self.current_goal:
                m = PoseStamped()
                m.header.frame_id = "map"
                m.header.stamp = rospy.Time.now()
                m.pose.position.x = self.current_goal[0]
                m.pose.position.y = self.current_goal[1]
                m.pose.position.z = 0.0
                m.pose.orientation.w = 1.0
                self.pub.publish(m)
            rate.sleep()


if __name__ == "__main__":
    FrontierExplorer().run()
