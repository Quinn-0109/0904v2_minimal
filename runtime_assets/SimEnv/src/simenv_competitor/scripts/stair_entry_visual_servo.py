#!/usr/bin/env python3
"""Independent visual-servo mission: return, face the stair, enter its approach area.

This is a top-level sideband mission.  It reads the RGB-D stair detector and
FAST-LIO odometry and sends only the existing ``/exploration_goal`` interface.
It never publishes ``/cmd_vel``, ``/joy``, Gazebo truth, or layout coordinates.

Unlike the previous one-shot goal, the approach is closed by the latest camera
measurement: drive a short leg towards the detected floor patch, reacquire the
first riser, then repeat.  This prevents an old visual estimate from carrying
the robot past the actual entrance.
"""
import json
import math
import os
import time

import rospy
from geometry_msgs.msg import PoseStamped, Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import String


def yaw(q):
    return math.atan2(2.0 * (q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))


def quat(a):
    return Quaternion(0.0, 0.0, math.sin(a/2.0), math.cos(a/2.0))


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class StairEntryServo:
    def __init__(self):
        self.frame = rospy.get_param('~frame_id', 'camera_init')
        self.output = rospy.get_param('~output_dir', '/tmp')
        self.home = (float(rospy.get_param('~home_x', 0.55)),
                     float(rospy.get_param('~home_y', 1.15)))
        self.goal_timeout = float(rospy.get_param('~goal_timeout', 120.0))
        self.detect_timeout = float(rospy.get_param('~detect_timeout', 20.0))
        self.max_legs = int(rospy.get_param('~max_approach_legs', 5))
        self.max_leg = float(rospy.get_param('~max_leg_m', 0.45))
        self.min_leg = float(rospy.get_param('~min_leg_m', 0.20))
        self.final_standoff = float(rospy.get_param('~final_standoff_m', 0.40))
        self.executor_tolerance = float(rospy.get_param('~executor_tolerance_m', 0.25))
        self.min_command_standoff = float(rospy.get_param('~min_command_standoff_m', 0.15))
        self.align_tol = float(rospy.get_param('~align_tol_rad', 0.16))
        self.turn_speed = float(rospy.get_param('~turn_speed_rad_s', 0.16))
        self.turn_chunk = float(rospy.get_param('~turn_chunk_rad', 0.28))
        self.observe_scan_steps = int(rospy.get_param('~observe_scan_steps', 12))
        self.observe_scan_direction = float(rospy.get_param('~observe_scan_direction', -1.0))
        self.landmark_samples = int(rospy.get_param('~landmark_samples', 5))
        self.landmark_max_spread = float(rospy.get_param('~landmark_max_spread_m', 0.65))
        self.minimum_progress = float(rospy.get_param('~minimum_progress_m', 0.06))
        self.resume_landmark_x = rospy.get_param('~resume_landmark_x', None)
        self.resume_landmark_y = rospy.get_param('~resume_landmark_y', None)
        self.odom = None; self.imu = None; self.det = None
        self.goal_result = None; self.scan_result = None; self.records = []
        self.goal_pub = rospy.Publisher('/exploration_goal', PoseStamped, queue_size=1)
        self.scan_pub = rospy.Publisher('/simenv/local_rescan_request', String, queue_size=1)
        rospy.Subscriber('/Odometry', Odometry, self.on_odom, queue_size=20)
        rospy.Subscriber('/trunk_imu', Imu, self.on_imu, queue_size=20)
        rospy.Subscriber('/simenv/stair_visual_detection', String, self.on_det, queue_size=20)
        rospy.Subscriber('/simenv/goal_execution_result', String, self.on_goal, queue_size=10)
        rospy.Subscriber('/simenv/local_rescan_result', String, self.on_scan, queue_size=10)

    def on_odom(self, m):
        p=m.pose.pose.position; self.odom=(p.x,p.y,p.z,yaw(m.pose.pose.orientation))
    def on_imu(self, m): self.imu=yaw(m.orientation)
    def on_det(self, m):
        try: self.det=json.loads(m.data)
        except Exception: pass
    def on_goal(self, m):
        try: self.goal_result=json.loads(m.data)
        except Exception: pass
    def on_scan(self, m):
        try: self.scan_result=json.loads(m.data)
        except Exception: pass

    def ready(self):
        until=time.time()+60.0
        while not rospy.is_shutdown() and time.time()<until:
            if self.odom and self.imu is not None and self.goal_pub.get_num_connections() and self.scan_pub.get_num_connections(): return True
            rospy.sleep(.1)
        return False

    def stable(self):
        d=self.det
        return bool(isinstance(d,dict) and d.get('detected_stable') and
                    isinstance(d.get('first_step_camera_point'),list) and
                    isinstance(d.get('entry_camera_point'),list) and
                    d.get('depth',{}).get('available') and d.get('depth',{}).get('ok'))

    def detection(self):
        until=time.time()+self.detect_timeout
        while not rospy.is_shutdown() and time.time()<until:
            if self.stable(): return self.det
            rospy.sleep(.1)
        return None

    def send_goal(self, name, x, y, a):
        self.goal_result=None
        m=PoseStamped();m.header.stamp=rospy.Time.now();m.header.frame_id=self.frame
        m.pose.position.x=x;m.pose.position.y=y;m.pose.orientation=quat(a)
        self.goal_pub.publish(m)
        until=time.time()+self.goal_timeout
        while not rospy.is_shutdown() and time.time()<until:
            r=self.goal_result; g=r.get('goal',{}) if isinstance(r,dict) else {}
            if isinstance(r,dict) and r.get('success') is not None and abs(g.get('x',1e9)-x)<.06 and abs(g.get('y',1e9)-y)<.06:
                self.records.append({'name':name,'goal':[x,y,a],'odom':self.odom,'result':r})
                return bool(r['success'])
            rospy.sleep(.1)
        self.records.append({'name':name,'goal':[x,y,a],'odom':self.odom,'result':{'success':False,'reason':'timeout'}})
        return False

    def face_bearing(self, bearing, label):
        """Use the existing bounded local-rescan interface for an in-place turn.

        Goal execution otherwise turns and translates together, which is what
        carried prior approaches into the wall.  Here every translation is
        preceded by a measured body alignment to the latest RGB-D ground ray.
        """
        desired=wrap(self.imu-bearing); first_sign=None
        for i in range(8):
            remain=wrap(desired-self.imu)
            if abs(remain)<=self.align_tol:
                self.records.append({'name':label,'success':True,'target_yaw':desired,'end_yaw':self.imu,'bearing':bearing})
                return True
            sign=1.0 if remain>0 else -1.0
            if first_sign is None: first_sign=sign
            if sign != first_sign: return False  # never visibly turn back
            rid='%s_%02d'%(label,i); self.scan_result=None
            self.scan_pub.publish(String(data=json.dumps({'request_id':rid,'angle_rad':min(self.turn_chunk,max(.10,abs(remain))), 'angular_speed':self.turn_speed,'direction':sign,'timeout_sec':10.0})))
            until=time.time()+15
            while not rospy.is_shutdown() and time.time()<until:
                if isinstance(self.scan_result,dict) and self.scan_result.get('request_id')==rid:
                    if not self.scan_result.get('success'): return False
                    break
                rospy.sleep(.1)
            else: return False
        return False

    def face_first_step(self, d):
        sx,_,sz=d['first_step_camera_point']
        return self.face_bearing(math.atan2(sx,sz), 'face_first_step')

    def scan_for_stair(self):
        """Single-direction observation scan; stop on the first stable stair."""
        d=self.detection()
        if d is not None:
            return d
        for i in range(self.observe_scan_steps):
            rid='stair_observe_%02d'%i; self.scan_result=None
            self.scan_pub.publish(String(data=json.dumps({
                'request_id':rid, 'angle_rad':self.turn_chunk,
                'angular_speed':self.turn_speed,
                'direction':self.observe_scan_direction, 'timeout_sec':10.0})))
            until=time.time()+15
            while not rospy.is_shutdown() and time.time()<until:
                if isinstance(self.scan_result,dict) and self.scan_result.get('request_id')==rid:
                    self.records.append({'name':rid,'direction':self.observe_scan_direction,'result':self.scan_result,'imu_yaw':self.imu})
                    if not self.scan_result.get('success'):
                        return None
                    break
                rospy.sleep(.1)
            else:
                return None
            d=self.detection()
            if d is not None:
                self.records.append({'name':'stair_observed','scan_index':i,'detection':d,'imu_yaw':self.imu})
                return d
        return None

    def candidate(self, d, leg):
        # Camera x is right, z is forward.  The entry pixel is a *ground* pixel
        # before the detected first riser; the step pixel is never used as goal.
        right, _down, forward = [float(v) for v in d['entry_camera_point']]
        rng=math.hypot(right,forward)
        scale=leg/max(rng,1e-6); forward*=scale; right*=scale
        ox,oy,_oz,myaw=self.odom
        x=ox+forward*math.cos(myaw)+right*math.sin(myaw)
        y=oy+forward*math.sin(myaw)-right*math.cos(myaw)
        return x,y,math.atan2(y-oy,x-ox),rng,forward,right

    def first_step_map(self, d):
        """Transform the detected first-riser point into camera_init/map."""
        right, _down, forward = [float(v) for v in d['first_step_camera_point']]
        ox,oy,_oz,myaw=self.odom
        return (ox + forward*math.cos(myaw) + right*math.sin(myaw),
                oy + forward*math.sin(myaw) - right*math.cos(myaw))

    def lock_first_step_landmark(self):
        """Collect a stationary visual landmark before translating.

        Entry-floor pixels vary with perspective.  The first riser itself is a
        physical landmark, so lock its median map position while the robot is
        not moving and reject a high-spread association.
        """
        samples=[]; until=time.time()+self.detect_timeout
        while not rospy.is_shutdown() and time.time()<until and len(samples)<self.landmark_samples:
            d=self.detection()
            if d is not None:
                samples.append(self.first_step_map(d))
            rospy.sleep(.12)
        if len(samples)<3:
            return None
        # RGB lines may offer two stair-like hypotheses.  Keep the largest
        # mutually consistent cluster; do not average two physical locations.
        clusters=[]
        for seed in samples:
            group=[p for p in samples if math.hypot(p[0]-seed[0],p[1]-seed[1]) <= self.landmark_max_spread]
            clusters.append(group)
        group=max(clusters,key=len)
        if len(group)<3:
            self.records.append({'name':'first_step_landmark','samples':samples,'reason':'no_three_frame_cluster'})
            return None
        xs=sorted(x for x,_ in group); ys=sorted(y for _,y in group)
        p=(xs[len(xs)//2],ys[len(ys)//2])
        spread=max(math.hypot(x-p[0],y-p[1]) for x,y in group)
        self.records.append({'name':'first_step_landmark','samples':samples,'inlier_samples':group,'landmark':p,'spread':spread})
        return p if spread<=self.landmark_max_spread else None

    def face_map_point(self, x, y, label):
        ox,oy,_oz,myaw=self.odom
        dx,dy=x-ox,y-oy
        # map displacement -> instantaneous body forward/right bearing.
        forward=math.cos(myaw)*dx+math.sin(myaw)*dy
        right=math.sin(myaw)*dx-math.cos(myaw)*dy
        return self.face_bearing(math.atan2(right,forward),label)

    def save(self, ok, reason):
        os.makedirs(self.output,exist_ok=True)
        with open(os.path.join(self.output,'stair_entry_visual_servo.json'),'w') as f:
            json.dump({'success':ok,'reason':reason,'records':self.records,'odom':self.odom,'imu_yaw':self.imu},f,indent=2)

    def run(self):
        if not self.ready(): return self.save(False,'not_ready')
        if self.resume_landmark_x is not None and self.resume_landmark_y is not None:
            landmark=(float(self.resume_landmark_x),float(self.resume_landmark_y))
            self.records.append({'name':'resume_visual_landmark','landmark':landmark,
                                 'source':'previous_visual_landmark_record'})
        else:
            if not self.send_goal('return_observation',self.home[0],self.home[1],0.0): return self.save(False,'return_failed')
            d=self.scan_for_stair()
            if d is None: return self.save(False,'no_stair_at_observation')
            landmark=self.lock_first_step_landmark()
            if landmark is None: return self.save(False,'first_step_landmark_not_stable')
        if not self.face_map_point(landmark[0], landmark[1], 'face_first_step'): return self.save(False,'align_failed')
        previous_distance=None
        for i in range(self.max_legs):
            ox,oy,_oz,_a=self.odom
            distance=math.hypot(landmark[0]-ox,landmark[1]-oy)
            self.records.append({'name':'entry_measurement_%02d'%i,'landmark_range':distance,'landmark':landmark,'odom':self.odom})
            if distance <= self.final_standoff + .08:
                return self.save(True,'stair_approach_area_reached')
            if previous_distance is not None and distance > previous_distance-self.minimum_progress:
                return self.save(False,'landmark_distance_not_decreasing')
            previous_distance=distance
            if not self.face_map_point(landmark[0], landmark[1], 'face_entry_%02d'%i):
                return self.save(False,'entry_alignment_failed')
            # Advance a bounded part of the *locked* landmark ray.  This does
            # not chase a changing floor pixel and never commands past the
            # retained first-step standoff.
            leg=min(self.max_leg,max(self.min_leg,distance-self.final_standoff))
            ratio=leg/max(distance,1e-6)
            x=ox+(landmark[0]-ox)*ratio; y=oy+(landmark[1]-oy)*ratio
            a=math.atan2(y-oy,x-ox)
            self.records.append({'name':'entry_leg_candidate_%02d'%i,'goal':[x,y,a],'landmark_range':distance,'leg':leg})
            if not self.send_goal('entry_leg_%02d'%i,x,y,a): return self.save(False,'entry_leg_failed')
        return self.save(False,'approach_leg_limit')


if __name__=='__main__':
    rospy.init_node('simenv_stair_entry_visual_servo')
    StairEntryServo().run()
