#!/usr/bin/env python3
# ponytail: passive recorder for TARE-driven runs. Subscribes only (never drives /cmd_vel).
# Accumulates /registered_scan -> binary PCD map, records /Odometry_gazebo -> trajectory.csv,
# and writes per-region coverage summary from layout_metadata.json. Dump every dump_period
# and on shutdown so a killed run still leaves a recent map.
import rospy, sys, os, csv, json
import numpy as np
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry

_DT = {1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2', 5: 'i4', 6: 'u4', 7: 'f4', 8: 'f8'}
VOXEL = 0.1  # m

class Recorder:
    def __init__(self, out_dir, lidar, odom, metadata, dump_period):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.meta_path = metadata
        self.clouds = []          # list of (N,3) float32 arrays
        self.traj = []            # (sim_t, x, y, z)
        rospy.Subscriber(lidar, PointCloud2, self.on_scan, queue_size=1)
        rospy.Subscriber(odom, Odometry, self.on_odom, queue_size=200)
        rospy.Timer(rospy.Duration(dump_period), lambda e: self.dump(final=False))
        rospy.on_shutdown(lambda: self.dump(final=True))

    def on_scan(self, msg):
        fs = [(f.name, _DT[f.datatype], f.offset) for f in msg.fields if f.name in ('x', 'y', 'z')]
        if len(fs) < 3:
            return
        dt = np.dtype({'names': [f[0] for f in fs], 'formats': [f[1] for f in fs],
                       'offsets': [f[2] for f in fs], 'itemsize': msg.point_step})
        n = msg.width * msg.height
        try:
            arr = np.frombuffer(msg.data, dtype=dt, count=n)
            xyz = np.stack([np.array(arr['x']), np.array(arr['y']), np.array(arr['z'])], axis=1)
            finite = np.all(np.isfinite(xyz), axis=1)
            self.clouds.append(xyz[finite].astype(np.float32))
        except Exception as e:
            rospy.logwarn_throttle(30, "scan parse failed: %s" % e)

    def on_odom(self, msg):
        t = msg.header.stamp.to_sec() if msg.header.stamp.to_sec() > 0 else rospy.Time.now().to_sec()
        p = msg.pose.pose.position
        self.traj.append((t, p.x, p.y, p.z))

    def _merged_xyz(self):
        if not self.clouds:
            return np.zeros((0, 3), np.float32)
        m = np.concatenate(self.clouds, axis=0)
        if VOXEL > 0 and len(m) > 0:                       # voxel downsample
            key = np.floor(m / VOXEL).astype(np.int64)
            _, idx = np.unique(key, axis=0, return_index=True)
            m = m[idx]
        return m

    def _regions(self):
        regions = {}
        try:
            meta = json.load(open(self.meta_path))
        except Exception:
            return regions
        def add(name, b):
            if isinstance(b, dict) and all(k in b for k in ('xmin', 'xmax', 'ymin', 'ymax')):
                regions[name] = (float(b['xmin']), float(b['xmax']), float(b['ymin']), float(b['ymax']))
        for key in ('lobby', 'corridor', 'elevator'):
            v = meta.get(key)
            if isinstance(v, dict):
                add(key, v.get('bounds', v))
        rooms = meta.get('rooms')
        if isinstance(rooms, list):
            for i, r in enumerate(rooms):
                if isinstance(r, dict):
                    add('room_%d' % i, r.get('bounds', r))
        return regions

    def dump(self, final=False):
        tag = "FINAL" if final else "periodic"
        try:
            xyz = self._merged_xyz()
            # binary PCD
            with open(os.path.join(self.out_dir, 'map.pcd'), 'wb') as f:
                n = len(xyz)
                f.write(b"# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\n"
                        b"SIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n")
                f.write(("WIDTH %d\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS %d\nDATA binary\n" % (n, n)).encode())
                xyz.astype('<f4').tofile(f)
            # trajectory csv
            with open(os.path.join(self.out_dir, 'trajectory.csv'), 'w', newline='') as f:
                w = csv.writer(f); w.writerow(['sim_time', 'x', 'y', 'z']); w.writerows(self.traj)
            # coverage summary
            traj = np.array([[r[1], r[2]] for r in self.traj], dtype=np.float64) if self.traj else np.zeros((0, 2))
            t_span = (self.traj[-1][0] - self.traj[0][0]) if len(self.traj) > 1 else 0.0
            path = float(np.sum(np.hypot(np.diff(traj[:, 0]), np.diff(traj[:, 1])))) if len(traj) > 1 else 0.0
            lines = ["exploration_sim_time_sec=%.2f" % t_span, "trajectory_path_m=%.2f" % path,
                     "trajectory_samples=%d" % len(self.traj), "map_points_downsampled=%d" % len(xyz), ""]
            regions = self._regions()
            for name, (xmin, xmax, ymin, ymax) in sorted(regions.items()):
                tin = np.where((traj[:, 0] >= xmin) & (traj[:, 0] <= xmax) &
                               (traj[:, 1] >= ymin) & (traj[:, 1] <= ymax))[0] if len(traj) else []
                tin_path = float(np.sum(np.hypot(np.diff(traj[tin, 0]), np.diff(traj[tin, 1])))) if len(tin) > 1 else 0.0
                pts_in = int(np.sum((xyz[:, 0] >= xmin) & (xyz[:, 0] <= xmax) &
                                    (xyz[:, 1] >= ymin) & (xyz[:, 1] <= ymax))) if len(xyz) else 0
                lines.append("region=%s entered=%s traj_samples=%d traj_path_m=%.2f map_points=%d bounds=x[%.1f,%.1f]_y[%.1f,%.1f]"
                             % (name, bool(len(tin) > 0), len(tin), tin_path, pts_in, xmin, xmax, ymin, ymax))
            with open(os.path.join(self.out_dir, 'summary.txt'), 'w') as f:
                f.write("\n".join(lines) + "\n")
            rospy.loginfo("passive_recorder %s dump: map=%dpts traj=%d path=%.1fm regions=%d" %
                          (tag, len(xyz), len(self.traj), path, len(regions)))
        except Exception as e:
            rospy.logerr("dump failed: %s" % e)

def main():
    out = sys.argv[1]
    lidar = sys.argv[2] if len(sys.argv) > 2 else '/registered_scan'
    odom = sys.argv[3] if len(sys.argv) > 3 else '/Odometry_gazebo'
    meta = sys.argv[4] if len(sys.argv) > 4 else '/workspace/SimEnv/generated_building/layout_metadata.json'
    dp = float(sys.argv[5]) if len(sys.argv) > 5 else 60.0
    rospy.init_node('passive_map_recorder', anonymous=True)
    Recorder(out, lidar, odom, meta, dp)
    rospy.loginfo("passive_recorder started -> %s (lidar=%s odom=%s meta=%s dump=%ss)" % (out, lidar, odom, meta, dp))
    rospy.spin()

if __name__ == '__main__':
    main()
