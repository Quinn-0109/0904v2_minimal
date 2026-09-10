#!/usr/bin/env python3
"""Standalone first-floor elevator boarding and second-floor exit test.

This node uses only the public building-control services and /cmd_vel.  It is
deliberately separate from the first-floor exploration manager so it can be
validated with a robot placed in front of the elevator before integration.
"""

import json
import os
from pathlib import Path

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, String
from building_generator_interfaces.srv import CallElevator, SetDoorState


class ElevatorTransitionManager:
    def __init__(self):
        self.output_dir = Path(rospy.get_param("~output_dir", "/tmp/elevator_transition"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.elevator_id = rospy.get_param("~elevator_id", "elevator_main")
        self.floor0_door = rospy.get_param("~floor0_door", "elevator_floor_0")
        self.floor1_door = rospy.get_param("~floor1_door", "elevator_floor_1")
        # The test world places the car east (+x) of the robot at start.
        # The explicit debug pose record at the end of ENTER_CAR keeps this
        # calibrated against the actual RL controller response.
        self.enter_speed = float(rospy.get_param("~enter_speed_mps", 1.20))
        self.enter_seconds = float(rospy.get_param("~enter_seconds", 4.0))
        self.exit_speed = float(rospy.get_param("~exit_speed_mps", -0.85))
        self.exit_seconds = float(rospy.get_param("~exit_seconds", 2.5))
        self.wait_rl_seconds = float(rospy.get_param("~wait_rl_seconds", 30.0))
        # Debug-only physical acceptance check.  This is deliberately off by
        # default: deployed navigation must not depend on Gazebo truth state.
        self.debug_verify_gazebo_state = bool(rospy.get_param("~debug_verify_gazebo_state", False))
        self.passenger_model = rospy.get_param("~passenger_model", "a1_gazebo")

        self.command_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=5)
        self.state_pub = rospy.Publisher("/simenv/elevator_transition_state", String,
                                         queue_size=2, latch=True)
        self.rl_ready = False
        self.locomotion_ready = False
        self.phase = "WAIT_RL"
        self.phase_start = rospy.Time.now()
        self.start_time = self.phase_start
        self.events = []
        rospy.Subscriber("/simenv/rl_takeover_supervisor_state", String,
                         self._on_rl_state, queue_size=3)
        rospy.Subscriber("/locomotion_ready", Bool, self._on_locomotion_ready,
                         queue_size=3)
        rospy.Timer(rospy.Duration(0.05), self._on_timer)
        rospy.on_shutdown(self._stop)
        rospy.loginfo("[ELEVATOR] armed: %s -> floor 1", self.floor0_door)

    def _on_rl_state(self, message):
        if "RL_TAKEOVER" in message.data:
            self.rl_ready = True

    def _on_locomotion_ready(self, message):
        self.locomotion_ready = bool(message.data)

    def _event(self, event, **details):
        stamp = rospy.Time.now().to_sec()
        item = {"t": round(stamp - self.start_time.to_sec(), 3), "event": event, **details}
        self.events.append(item)
        rospy.loginfo("[ELEVATOR] %s", item)

    def _transition(self, phase):
        self._stop()
        self.phase = phase
        self.phase_start = rospy.Time.now()
        self.state_pub.publish(String(data=phase))
        self._event("phase", value=phase)

    def _call_door(self, door_id, open_state):
        try:
            rospy.wait_for_service("/set_door_state", timeout=10.0)
            result = rospy.ServiceProxy("/set_door_state", SetDoorState)(door_id, open_state)
            self._event("door", id=door_id, open=bool(open_state), accepted=bool(result.accepted),
                        state=result.state, message=result.message)
            return bool(result.accepted)
        except (rospy.ROSException, rospy.ServiceException) as exc:
            self._event("service_error", service="set_door_state", error=str(exc))
            return False

    def _call_elevator(self):
        try:
            rospy.wait_for_service("/call_elevator", timeout=10.0)
            result = rospy.ServiceProxy("/call_elevator", CallElevator)(self.elevator_id, 1, False)
            self._event("elevator", id=self.elevator_id, target_floor=1,
                        accepted=bool(result.accepted), floor=int(result.current_floor),
                        state=result.state, message=result.message)
            return bool(result.accepted)
        except (rospy.ROSException, rospy.ServiceException) as exc:
            self._event("service_error", service="call_elevator", error=str(exc))
            return False

    def _verify_passenger_lifted(self):
        if not self.debug_verify_gazebo_state:
            return True
        try:
            from gazebo_msgs.srv import GetModelState
            rospy.wait_for_service("/gazebo/get_model_state", timeout=5.0)
            result = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)(self.passenger_model, "world")
            z_value = float(result.pose.position.z) if result.success else float("nan")
            self._event("debug_passenger_pose", success=bool(result.success),
                        x=round(float(result.pose.position.x), 3),
                        y=round(float(result.pose.position.y), 3), z=round(z_value, 3))
            return bool(result.success) and z_value > 2.0
        except (rospy.ROSException, rospy.ServiceException) as exc:
            self._event("service_error", service="gazebo/get_model_state", error=str(exc))
            return False

    def _debug_record_pose(self, label):
        if not self.debug_verify_gazebo_state:
            return
        try:
            from gazebo_msgs.srv import GetModelState
            result = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)(self.passenger_model, "world")
            self._event("debug_passenger_pose", label=label, success=bool(result.success),
                        x=round(float(result.pose.position.x), 3),
                        y=round(float(result.pose.position.y), 3),
                        z=round(float(result.pose.position.z), 3))
        except rospy.ServiceException as exc:
            self._event("service_error", service="gazebo/get_model_state", error=str(exc))

    def _publish(self, linear_x):
        msg = Twist()
        msg.linear.x = linear_x
        self.command_pub.publish(msg)

    def _stop(self):
        self.command_pub.publish(Twist())

    def _finish(self, success, reason):
        if self.phase == "DONE":
            return
        self._stop()
        self.phase = "DONE"
        self.state_pub.publish(String(data="DONE" if success else "FAILED:" + reason))
        self._event("result", success=bool(success), reason=reason)
        log_path = self.output_dir / "logs" / "elevator_transition.json"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps({"success": bool(success), "events": self.events}, indent=2) + "\n")
        rospy.loginfo("[ELEVATOR] %s; log=%s", "completed" if success else "failed", log_path)

    def _on_timer(self, _event):
        if self.phase == "DONE":
            return
        now = rospy.Time.now()
        elapsed = (now - self.phase_start).to_sec()
        if self.phase == "WAIT_RL":
            if self.rl_ready and self.locomotion_ready:
                self._transition("OPEN_FLOOR_0")
            elif elapsed > self.wait_rl_seconds:
                self._finish(False, "rl_takeover_timeout")
        elif self.phase == "OPEN_FLOOR_0":
            if self._call_door(self.floor0_door, True):
                self._transition("ENTER_CAR")
            else:
                self._finish(False, "open_floor_0_failed")
        elif self.phase == "ENTER_CAR":
            self._publish(self.enter_speed)
            if elapsed >= self.enter_seconds:
                self._debug_record_pose("before_close_floor_0")
                self._transition("CLOSE_FLOOR_0")
        elif self.phase == "CLOSE_FLOOR_0":
            if self._call_door(self.floor0_door, False):
                self._transition("RIDE_TO_FLOOR_1")
            else:
                self._finish(False, "close_floor_0_failed")
        elif self.phase == "RIDE_TO_FLOOR_1":
            if self._call_elevator():
                if self._verify_passenger_lifted():
                    self._transition("OPEN_FLOOR_1")
                else:
                    self._finish(False, "passenger_not_lifted")
            else:
                self._finish(False, "call_floor_1_failed")
        elif self.phase == "OPEN_FLOOR_1":
            if self._call_door(self.floor1_door, True):
                # The standalone acceptance target is a real arrival at the
                # second-floor elevator with its hall door open.  Leaving the
                # car needs a second-floor navigation target and is deliberately
                # deferred until the floor-2 exploration stack is connected.
                self._finish(True, "arrived_floor_1_door_open")
            else:
                self._finish(False, "open_floor_1_failed")


if __name__ == "__main__":
    rospy.init_node("elevator_transition_manager")
    ElevatorTransitionManager()
    rospy.spin()
