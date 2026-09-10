#!/usr/bin/env python3
"""State-gated FixedStand -> RL takeover via /joy."""

import rospy
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool, String


class ControllerModeBootstrap:
    def __init__(self):
        self._publisher = rospy.Publisher("/joy", Joy, queue_size=3)
        self._stand_after = float(rospy.get_param("~stand_after_seconds", 0.5))
        self._max_wait = float(rospy.get_param("~fixed_stand_max_duration_sec", 15.0))
        self._rl_repeat_seconds = float(rospy.get_param("~rl_repeat_seconds", 40.0))
        # A dedicated motion driver may publish button-3 continuously after
        # take-over.  In that case this bootstrap must relinquish /joy instead
        # of later issuing a stale Passive command.
        self._release_after_rl_request = bool(rospy.get_param(
            "~release_after_rl_request", False))
        self._phase = "wait_stand"
        self._stand_sent = False
        # Wait for /clock so elapsed is not inflated by Time(0)->sim jump.
        if rospy.get_param("/use_sim_time", False):
            while not rospy.is_shutdown() and rospy.Time.now().to_sec() < 1.0:
                rospy.sleep(0.05)
        self._start = rospy.Time.now()
        self._logged = False
        self._fixed_stand_ready = False
        self._locomotion_ready = False
        self._stand_state_started = None
        self._failed = False
        self._state_pub = rospy.Publisher(
            "/simenv/rl_takeover_supervisor_state", String, queue_size=2, latch=True)
        rospy.Subscriber("/fixed_stand_ready", Bool, self._on_stand_ready, queue_size=2)
        rospy.Subscriber("/locomotion_ready", Bool,
                         self._on_locomotion_ready, queue_size=2)
        rospy.Timer(rospy.Duration(0.1), self._on_timer)
        rospy.loginfo(
            "Bootstrap armed: stand@%.1fs state-gated RL max_wait=%.1fs repeat=%.1fs start_t=%.1f",
            self._stand_after,
            self._max_wait,
            self._rl_repeat_seconds,
            self._start.to_sec(),
        )

    def _on_stand_ready(self, message):
        if self._stand_state_started is None:
            self._stand_state_started = rospy.Time.now()
        # FixedStand naturally publishes false again after handing over to RL.
        # Bootstrap only needs to know whether the initial stand phase ever
        # became stable; do not let that later transition re-arm its timeout
        # and override a running RL stair policy with Passive.
        self._fixed_stand_ready = self._fixed_stand_ready or bool(message.data)

    def _on_locomotion_ready(self, message):
        self._locomotion_ready = bool(message.data)

    def _publish_button(self, index):
        message = Joy()
        message.header.stamp = rospy.Time.now()
        message.axes = [0.0] * 8
        message.buttons = [0] * 12
        message.buttons[index] = 1
        self._publisher.publish(message)

    def _on_timer(self, _event):
        if self._phase == "done" or self._failed:
            return
        elapsed = (rospy.Time.now() - self._start).to_sec()
        if elapsed < self._stand_after:
            return
        if not self._stand_sent:
            self._publish_button(1)  # fixed stand once
            self._stand_sent = True
            self._phase = "WAIT_STABLE"
            self._state_pub.publish(String(data=self._phase))
            return
        if not self._fixed_stand_ready:
            # SIMENV_REPEAT_FIXED_STAND: retry until junior_ctrl acknowledges it.
            self._publish_button(1)
            stand_elapsed = ((rospy.Time.now() - self._stand_state_started).to_sec()
                             if self._stand_state_started is not None else 0.0)
            if self._stand_state_started is not None and stand_elapsed >= self._max_wait:
                self._publish_button(0)  # passive; never force RL after timeout
                self._failed = True
                self._phase = "FAILED"
                self._state_pub.publish(String(data="FAILED:fixed_stand_stability_timeout"))
                rospy.logerr("FixedStand did not become state-stable within %.1fs; RL blocked",
                             self._max_wait)
            return
        if elapsed < self._max_wait + self._rl_repeat_seconds:
            if self._release_after_rl_request and self._locomotion_ready:
                self._phase = "done"
                self._state_pub.publish(String(data="READY:locomotion_ready"))
                rospy.loginfo(
                    "Bootstrap observed locomotion_ready; releasing /joy.")
                return
            self._publish_button(3)  # RL /cmd_vel
            self._phase = "RL_TAKEOVER"
            self._state_pub.publish(String(data=self._phase))
            if not self._logged:
                rospy.loginfo("Requesting RL /cmd_vel through /joy (repeating).")
                self._logged = True
            return
        self._phase = "done"
        rospy.loginfo("Bootstrap finished; RL should be active.")


if __name__ == "__main__":
    rospy.init_node("controller_mode_bootstrap")
    ControllerModeBootstrap()
    rospy.spin()
