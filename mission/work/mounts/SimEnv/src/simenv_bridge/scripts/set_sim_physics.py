#!/usr/bin/env python3
# ponytail: one-shot — force Gazebo physics to RTF~1 on every explore start.
# Gazebo Classic defaults to 1000Hz/0.001s -> RTF 0.067 on this heavy multi-floor world
# (physics is single-threaded). time_step 0.004 + rate 250 + iters 50->20 gives RTF~1.
# Kinematic driver sets the base pose directly, so looser physics is safe; exploration
# uses /registered_scan (geometry), independent of the physics solver. Revert via auto.sh
# defaults if a future change adds balance-critical dynamics.
import rospy
from gazebo_msgs.srv import SetPhysicsProperties, SetPhysicsPropertiesRequest
from geometry_msgs.msg import Vector3

rospy.init_node('set_sim_physics', anonymous=True)
try:
    rospy.wait_for_service('/gazebo/set_physics_properties', timeout=60)
except rospy.ROSException:
    rospy.logerr("set_sim_physics: /gazebo/set_physics_properties never came up; RTF stays at Gazebo default.")
    raise SystemExit(0)

sp = rospy.ServiceProxy('/gazebo/set_physics_properties', SetPhysicsProperties)
req = SetPhysicsPropertiesRequest()
req.time_step = 0.004
req.max_update_rate = 250.0
req.gravity = Vector3(0.0, 0.0, -9.8)
o = req.ode_config
o.auto_disable_bodies = False
o.sor_pgs_precon_iters = 0
o.sor_pgs_iters = 20
o.sor_pgs_w = 1.3
o.sor_pgs_rms_error_tol = 0.0
o.contact_surface_layer = 0.001
o.contact_max_correcting_vel = 100.0
o.cfm = 0.0
o.erp = 0.2
o.max_contacts = 20
try:
    resp = sp(req)
    rospy.loginfo("set_sim_physics: time_step=0.004 max_update_rate=250 sor_pgs_iters=20 -> success=%s (%s)"
                  % (resp.success, resp.status_message))
except Exception as e:
    rospy.logerr("set_sim_physics call failed: %s" % e)
