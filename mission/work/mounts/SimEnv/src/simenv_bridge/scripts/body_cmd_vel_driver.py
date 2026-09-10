#!/usr/bin/env python3
"""body_cmd_vel_driver: body-frame cmd_vel → Gazebo set_model_state (position + yaw).
For SCAN-Planner integration: SCAN-Planner publishes body-frame Twist
(linear.x=forward, linear.y=lateral, angular.z=yaw_rate). This driver integrates it
into world position + orientation (unlike cmd_vel_to_model_state which is world-only, max_wz=0)."""
import rospy, math
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool


def select_velocity(planner_velocity, scan_active, scan_velocity):
    return scan_velocity if scan_active else planner_velocity


def apply_linear_scale(velocity, scan_active, scale):
    if scan_active:
        return velocity
    return velocity[0] * scale, velocity[1] * scale, velocity[2]


def limit_planar_velocity(vx, vy, max_v):
    """Limit the planar vector norm, preserving direction."""
    norm = math.hypot(float(vx), float(vy))
    if norm <= float(max_v) or norm < 1e-9:
        return float(vx), float(vy)
    factor = float(max_v) / norm
    return float(vx) * factor, float(vy) * factor


def integration_dt(now, previous, rate_hz, max_dt=0.2):
    """Use simulation time so service-loop jitter does not reduce velocity."""
    if previous is None:
        return 1.0 / float(rate_hz)
    return max(0.0, min(float(max_dt), float(now) - float(previous)))


def should_initialize_pose(pose_initialized):
    return not bool(pose_initialized)


class BodyCmdVelDriver:
    def __init__(self):
        self.model_name = rospy.get_param('~model_name', 'a1_gazebo')
        self.rate_hz = rospy.get_param('~rate', 30)
        self.max_v = rospy.get_param('~max_v', 1.5)
        self.max_w = rospy.get_param('~max_w', 1.0)
        self.linear_scale = float(rospy.get_param('~linear_scale', 1.0))
        self.lock_z = rospy.get_param('~lock_z', 0.6)
        self.planner_cmd = Twist()
        self.scan_cmd = Twist()
        self.scan_active = False
        self.x = 0.0; self.y = 0.0; self.yaw = 1.5708
        self.pose_initialized = False
        self.last_t = None
        self.last_speed_log = None
        self.applied_pub = rospy.Publisher('/scanplanner/applied_cmd_vel', Twist, queue_size=10)
        rospy.Subscriber('/cmd_vel', Twist, self.cmd_cb, queue_size=1)
        rospy.Subscriber('/scanplanner/scan_cmd_vel', Twist, self.scan_cmd_cb, queue_size=1)
        rospy.Subscriber('/scanplanner/scan_active', Bool, self.scan_active_cb, queue_size=1)
        rospy.Subscriber('/Odometry_gazebo', Odometry, self.odom_cb, queue_size=10)
        rospy.wait_for_service('/gazebo/set_model_state', timeout=10)
        self.set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        rospy.loginfo("body_cmd_vel_driver: ready (model=%s max_v=%.1f max_w=%.1f)" % (self.model_name, self.max_v, self.max_w))

    def cmd_cb(self, msg):
        self.planner_cmd = msg

    def scan_cmd_cb(self, msg):
        self.scan_cmd = msg

    def scan_active_cb(self, msg):
        self.scan_active = bool(msg.data)

    def odom_cb(self, msg):
        if not should_initialize_pose(self.pose_initialized):
            return
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.x = p.x; self.y = p.y
        self.yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
        self.pose_initialized = True

    def run(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown() and not self.pose_initialized:
            rate.sleep()
        while not rospy.is_shutdown():
            now = rospy.get_time()
            dt = integration_dt(now, self.last_t, self.rate_hz)
            self.last_t = now
            planner = (self.planner_cmd.linear.x, self.planner_cmd.linear.y, self.planner_cmd.angular.z)
            scan = (self.scan_cmd.linear.x, self.scan_cmd.linear.y, self.scan_cmd.angular.z)
            selected = select_velocity(planner, self.scan_active, scan)
            selected = apply_linear_scale(selected, self.scan_active, self.linear_scale)
            vx, vy = limit_planar_velocity(selected[0], selected[1], self.max_v)
            wz = max(-self.max_w, min(self.max_w, selected[2]))
            applied = Twist()
            applied.linear.x = vx
            applied.linear.y = vy
            applied.angular.z = wz
            self.applied_pub.publish(applied)
            # integrate yaw + body→world position
            self.yaw += wz * dt
            wx = vx * math.cos(self.yaw) - vy * math.sin(self.yaw)
            wy = vx * math.sin(self.yaw) + vy * math.cos(self.yaw)
            self.x += wx * dt
            self.y += wy * dt
            # set model state (position + orientation)
            state = ModelState()
            state.model_name = self.model_name
            state.reference_frame = 'world'
            state.pose.position.x = self.x
            state.pose.position.y = self.y
            state.pose.position.z = self.lock_z
            state.pose.orientation.z = math.sin(self.yaw / 2.0)
            state.pose.orientation.w = math.cos(self.yaw / 2.0)
            try:
                self.set_state(state)
            except Exception:
                pass
            if self.last_speed_log is None or now - self.last_speed_log >= 5.0:
                rospy.loginfo(
                    "body_cmd_vel_driver: planar_cmd=%.3f m/s (limit=%.3f)",
                    math.hypot(vx, vy), self.max_v,
                )
                self.last_speed_log = now
            rate.sleep()


if __name__ == '__main__':
    rospy.init_node('body_cmd_vel_driver', anonymous=True)
    BodyCmdVelDriver().run()
