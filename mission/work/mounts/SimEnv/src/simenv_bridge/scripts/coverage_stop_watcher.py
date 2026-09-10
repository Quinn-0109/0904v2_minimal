#!/usr/bin/env python3
# ponytail: coverage-complete stop supervisor.
# Stops the TARE explore when EVERY room has been entered AND each has accumulated
# >= min_path meters of trajectory (1Hz de-jittered), sustained for `sustain` sec.
# A max_time sim-second safety guarantees termination even if a room is unreachable.
# Chosen over frontier-exhaustion (TARE's local frontier never sustains-0; it regenerates).
import rospy, sys, os, json, math, subprocess
from nav_msgs.msg import Odometry

def load_rooms(meta_path):
    m = json.load(open(meta_path)); f0 = m['floors'][0]
    rooms = []
    for i, r in enumerate(f0['rooms']):
        b = r['bounds']
        rooms.append(('room_%d' % i, b['x_min'], b['x_max'], b['y_min'], b['y_max']))
    return rooms

class Watcher:
    def __init__(self, meta_path, marker_path, min_path, sustain, max_time):
        self.rooms = load_rooms(meta_path)
        self.marker_path = marker_path
        self.min_path = min_path
        self.sustain = sustain
        self.max_time = max_time
        self.path = {n: 0.0 for n, _, _, _, _ in self.rooms}
        self.entered = {n: False for n, _, _, _, _ in self.rooms}
        self.last_sample_t = None
        self.last_xy = None
        self.start_t = None
        self.since_ok = None
        self.done = False
        rospy.Subscriber('/Odometry_gazebo', Odometry, self.on_odom, queue_size=200)
        rospy.Timer(rospy.Duration(1.0), self.tick)

    def on_odom(self, msg):
        t = msg.header.stamp.to_sec()
        if t <= 0:
            return
        if self.start_t is None:
            self.start_t = t
        if self.last_sample_t is None or (t - self.last_sample_t) >= 1.0:   # 1Hz de-jitter
            p = msg.pose.pose.position
            if self.last_xy is not None:
                seg = math.hypot(p.x - self.last_xy[0], p.y - self.last_xy[1])
                for n, x0, x1, y0, y1 in self.rooms:
                    if x0 <= p.x <= x1 and y0 <= p.y <= y1:
                        self.path[n] += seg
                        self.entered[n] = True
            self.last_xy = (p.x, p.y)
            self.last_sample_t = t

    def tick(self, _):
        if self.done or self.start_t is None:
            return
        now = rospy.Time.now().to_sec()
        elapsed_sim = (self.last_sample_t - self.start_t) if self.last_sample_t else 0.0
        all_in = all(self.entered.values())
        all_path = all(self.path[n] >= self.min_path for n, *_ in self.rooms)
        ok = all_in and all_path
        self.since_ok = now if (ok and self.since_ok is None) else (None if not ok else self.since_ok)
        fire_cov = self.since_ok is not None and (now - self.since_ok) >= self.sustain
        fire_time = elapsed_sim >= self.max_time
        rospy.loginfo_throttle(10, "coverage_watcher: elapsed=%.0fs entered=%d/%d path=%s min=%.0f  cov_fire=%s time_fire=%s"
                               % (elapsed_sim, sum(self.entered.values()), len(self.rooms),
                                  {n: round(self.path[n], 1) for n, *_ in self.rooms}, self.min_path, fire_cov, fire_time))
        if fire_cov or fire_time:
            self.trigger(fire_time, elapsed_sim)

    def trigger(self, by_time, elapsed):
        self.done = True
        msg = ("COVERAGE_STOP (%s) at sim=%.0fs: entered=%d/%d path=%s min=%.0f"
               % ("max_time" if by_time else "coverage_complete", elapsed, sum(self.entered.values()),
                  len(self.rooms), {n: round(self.path[n], 1) for n, *_ in self.rooms}, self.min_path))
        rospy.logwarn("coverage_watcher: " + str(msg))
        try:
            os.makedirs(os.path.dirname(self.marker_path), exist_ok=True)
            with open(self.marker_path, 'w') as f:
                f.write(str(msg) + "\n")
        except Exception as e:
            rospy.logerr("marker write failed: %s" % e)
        for cmd in (["pkill", "-f", "roslaunch simenv_bridge explore_simenv"],
                    ["pkill", "-f", "explore_simenv.launch"]):
            subprocess.call(cmd)
        rospy.signal_shutdown("coverage stop")

def main():
    meta = sys.argv[1] if len(sys.argv) > 1 else '/workspace/SimEnv/generated_building/layout_metadata.json'
    marker = sys.argv[2] if len(sys.argv) > 2 else '/workspace/SimEnv/results/coverage_run/COVERAGE_STOP'
    min_path = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
    sustain = float(sys.argv[4]) if len(sys.argv) > 4 else 15.0
    max_time = float(sys.argv[5]) if len(sys.argv) > 5 else 580.0
    rospy.init_node('coverage_stop_watcher', anonymous=True)
    rospy.loginfo("coverage_watcher: rooms from %s, min_path=%.0fm sustain=%ss max_time=%.0fs" %
                  (meta, min_path, sustain, max_time))
    Watcher(meta, marker, min_path, sustain, max_time)
    rospy.spin()

if __name__ == '__main__':
    main()
