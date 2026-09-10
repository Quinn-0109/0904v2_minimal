#!/usr/bin/env python3
# ponytail: exploration stop-condition supervisor.
# Stops the TARE explore when the LOCAL frontier is exhausted (sustained <= frontier_max)
# AND a coverage guard is met (robot actually traversed the building), so we never trip the
# pre-fix premature-exhaustion failure (frontier=0 while rooms 2/3 unseen).
#
# Signals (from /tmp/tare_explore.log TARE_DIAG lines):
#   "local_coverage uncovered=N frontier=M"  -> M is the local-horizon frontier count
# Coverage guard (from /Odometry_gazebo): path length and y-span must exceed thresholds.
#
# On trigger: writes FRONTIER_EXHAUSTED marker + kills the explore roslaunch (keeps sim).
import rospy, sys, os, re, time, subprocess, math
from nav_msgs.msg import Odometry

class Watcher:
    def __init__(self, log_path, marker_path, frontier_max, sustain_sec, path_min, yspan_min):
        self.log_path = log_path
        self.marker_path = marker_path
        self.frontier_max = frontier_max
        self.sustain_sec = sustain_sec
        self.path_min = path_min
        self.yspan_min = yspan_min
        self.frontier = None
        self.since_zero = None          # sim-time when frontier first went <= max
        self.last_xy = None
        self.path = 0.0
        self.ymin = 1e9; self.ymax = -1e9
        self.done = False
        rospy.Subscriber('/Odometry_gazebo', Odometry, self.on_odom, queue_size=200)
        rospy.Timer(rospy.Duration(1.0), self.tick)

    def on_odom(self, msg):
        p = msg.pose.pose.position
        if self.last_xy is not None:
            self.path += math.hypot(p.x - self.last_xy[0], p.y - self.last_xy[1])
        self.last_xy = (p.x, p.y)
        self.ymin = min(self.ymin, p.y); self.ymax = max(self.ymax, p.y)

    def _read_frontier(self):
        # ponytail: tail-follow the log; grab the last 'local_coverage ... frontier=N'
        try:
            with open(self.log_path) as f:
                f.seek(0, os.SEEK_END); size = f.tell(); f.seek(max(0, size - 200000))
                tail = f.read()
        except Exception:
            return self.frontier
        ms = re.findall(r'local_coverage uncovered=\d+ frontier=(\d+)', tail)
        return int(ms[-1]) if ms else self.frontier

    def tick(self, _):
        if self.done:
            return
        self.frontier = self._read_frontier()
        now = rospy.Time.now().to_sec()
        if self.frontier is not None and self.frontier <= self.frontier_max:
            if self.since_zero is None:
                self.since_zero = now
        else:
            self.since_zero = None       # not exhausted right now -> reset sustained timer
        yspan = self.ymax - self.ymin
        guard = (self.path >= self.path_min) and (yspan >= self.yspan_min)
        sustained = self.since_zero is not None and (now - self.since_zero) >= self.sustain_sec
        rospy.loginfo_throttle(10, "frontier_watcher: frontier=%s path=%.1fm(>=%s) yspan=%.1fm(>=%s) sustained_zero=%ss guard=%s"
                               % (self.frontier, self.path, self.path_min, yspan, self.yspan_min,
                                  ('%.0f' % (now - self.since_zero)) if self.since_zero else '0', guard and sustained))
        if sustained and guard:
            self.trigger(yspan)

    def trigger(self, yspan):
        self.done = True
        msg = ("FRONTIER_EXHAUSTED at sim=%.1f: frontier<=%d sustained %ss; path=%.1fm; yspan=%.1fm; ymin=%.1f ymax=%.1f"
               % (rospy.Time.now().to_sec(), self.frontier_max, self.sustain_sec, self.path, yspan, self.ymin, self.ymax))
        rospy.logwarn("frontier_watcher: " + msg)
        try:
            os.makedirs(os.path.dirname(self.marker_path), exist_ok=True)
            with open(self.marker_path, 'w') as f:
                f.write(msg + "\n")
        except Exception as e:
            rospy.logerr("marker write failed: %s" % e)
        # kill the explore roslaunch (keep sim + rosbag + recorder alive briefly)
        for cmd in (["pkill", "-f", "roslaunch simenv_bridge explore_simenv"],
                    ["pkill", "-f", "explore_simenv.launch"]):
            subprocess.call(cmd)
        rospy.signal_shutdown("frontier exhausted")

def main():
    log = sys.argv[1] if len(sys.argv) > 1 else '/tmp/tare_explore.log'
    marker = sys.argv[2] if len(sys.argv) > 2 else '/workspace/SimEnv/results/frontier_run/FRONTIER_EXHAUSTED'
    fmax = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    sustain = float(sys.argv[4]) if len(sys.argv) > 4 else 60.0
    pmin = float(sys.argv[5]) if len(sys.argv) > 5 else 50.0
    ysmin = float(sys.argv[6]) if len(sys.argv) > 6 else 25.0   # 4-room: robot must reach rooms 2/3
    rospy.init_node('frontier_stop_watcher', anonymous=True)
    rospy.loginfo("frontier_watcher: log=%s frontier<=%d sustain=%ss path>=%.0fm yspan>=%.1fm" %
                  (log, fmax, sustain, pmin, ysmin))
    Watcher(log, marker, fmax, sustain, pmin, ysmin)
    rospy.spin()

if __name__ == '__main__':
    main()
