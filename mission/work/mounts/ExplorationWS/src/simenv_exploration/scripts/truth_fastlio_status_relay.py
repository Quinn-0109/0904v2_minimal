#!/usr/bin/env python3
"""Truth-mode FAST-LIO status relay.

Without FAST-LIO the health statuses that downstream gates expect on
/simenv/fastlio_*_status never appear (laserMapping.cpp is the only
publisher).  In truth mode "registration" is a direct geometric transform
of Gazebo truth, so it is healthy by construction and the truth guard needs
no correction.  This node mirrors the FAST-LIO JSON payloads so the
F2/F3 handoff stabilization gates (which require a fresh healthy
registration status and an active F2 truth guard) pass immediately.

Payloads intentionally match publish_registration_status() and
publish_second_floor_truth_guard_status() in FAST_LIO/src/laserMapping.cpp.
"""

import json
import math

import rospy
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import String


class TruthFastlioStatusRelay:
    def __init__(self):
        self._model = str(rospy.get_param("~model_name", "a1_gazebo"))
        self._rate = float(rospy.get_param("~rate_hz", 5.0))
        self._position = (0.0, 0.0, 0.0)
        self._registration_pub = rospy.Publisher(
            "/simenv/fastlio_registration_status", String, queue_size=5)
        self._truth_guard_pub = rospy.Publisher(
            "/simenv/fastlio_f2_truth_guard_status", String, queue_size=5)
        rospy.Subscriber("/gazebo/model_states", ModelStates,
                         self._on_model_states, queue_size=4)
        rospy.Timer(rospy.Duration(1.0 / self._rate), self._on_timer)
        rospy.loginfo("Truth FAST-LIO status relay ready (%.1f Hz)",
                      self._rate)

    def _on_model_states(self, message):
        try:
            index = message.name.index(self._model)
        except ValueError:
            return
        p = message.pose[index].position
        if all(math.isfinite(v) for v in (p.x, p.y, p.z)):
            self._position = (p.x, p.y, p.z)

    def _on_timer(self, _event):
        if rospy.is_shutdown():
            return
        registration = {
            "healthy": True,
            "frozen": False,
            "effective_points": 100,
            "invalid_count": 0,
            "innovation_rejected": False,
            "planar_recovery_applied": False,
            "horizontal_step_m": 0.0,
            "allowed_horizontal_step_m": 0.0,
            "vertical_step_m": 0.0,
        }
        self._registration_pub.publish(String(
            data=json.dumps(registration)))
        x, y, z = self._position
        truth_guard = {
            "enabled": True,
            "f2_context": True,
            "truth_received": True,
            "truth_stale": False,
            "active": True,
            "truth_age_sec": 0.0,
            "correction_m": 0.0,
            "yaw_correction_rad": 0.0,
            "correction_count": 0,
            "maximum_correction_m": 0.0,
            "height_recovery_count": 0,
            "anchor_height_error_m": 0.0,
            "position_x": x,
            "position_y": y,
            "position_z": z,
        }
        self._truth_guard_pub.publish(String(data=json.dumps(truth_guard)))


if __name__ == "__main__":
    rospy.init_node("truth_fastlio_status_relay")
    TruthFastlioStatusRelay()
    rospy.spin()
