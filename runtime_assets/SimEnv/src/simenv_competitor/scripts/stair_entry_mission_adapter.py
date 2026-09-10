#!/usr/bin/env python3
"""Trigger/result adapter for the preserved visual stair-entry mission."""
import importlib.util
import json
import os
import time

import rospy
from std_msgs.msg import String

_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "stair_return_turn_entry.py")
_spec = importlib.util.spec_from_file_location("stair_return_turn_entry", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


class StairEntryMissionAdapter(_module.ReturnTurnEntry):
    def __init__(self):
        super().__init__()
        self.trigger_topic = str(rospy.get_param(
            "~mission_trigger_topic", "")).strip()
        self.trigger_token = str(rospy.get_param(
            "~mission_trigger_token", "")).strip()
        self.trigger_timeout = float(rospy.get_param(
            "~mission_trigger_timeout_sec", 900.0))
        self.ready_token = str(rospy.get_param(
            "~entry_ready_token", "STAIR_VISUAL_ENTRY_READY"))
        state_topic = str(rospy.get_param(
            "~entry_state_topic", "/simenv/stair_visual_entry_state"))
        self.triggered = not bool(self.trigger_topic)
        self.entry_state = rospy.Publisher(state_topic, String, queue_size=1,
                                           latch=True)
        if self.trigger_topic:
            rospy.Subscriber(self.trigger_topic, String, self.on_trigger,
                             queue_size=5)

    @staticmethod
    def matches(payload, token):
        expected = str(token or "").strip()
        actual = str(payload or "").strip()
        if actual == expected:
            return True
        try:
            decoded = json.loads(actual)
        except (TypeError, ValueError):
            return False
        return (isinstance(decoded, dict) and
                str(decoded.get("state", "")).strip() == expected)

    def on_trigger(self, message):
        if self.matches(message.data, self.trigger_token):
            self.triggered = True

    def wait_for_trigger(self):
        if self.triggered:
            return True
        self.entry_state.publish(String(data="WAITING_FOR_FLOOR_HANDOFF"))
        deadline = time.monotonic() + max(0.0, self.trigger_timeout)
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            if self.triggered:
                self.entry_state.publish(String(data="VISUAL_ENTRY_ACTIVE"))
                return True
            rospy.sleep(0.10)
        return False

    def save(self, ok, reason):
        super().save(ok, reason)
        if ok:
            self.entry_state.publish(String(data=self.ready_token))
        else:
            self.entry_state.publish(String(data=json.dumps({
                "state": "STAIR_VISUAL_ENTRY_FAILED", "reason": reason},
                sort_keys=True)))

    def run(self):
        if not self.wait_for_trigger():
            return self.save(False, "mission_trigger_timeout")
        return super().run()


def main():
    rospy.init_node("stair_entry_mission_adapter")
    StairEntryMissionAdapter().run()


if __name__ == "__main__":
    main()
