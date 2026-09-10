#!/usr/bin/env python3
"""Audit short in-place RGB-D sweeps made by the exploration manager.

Motion ownership deliberately remains with baseline_exploration_manager and
goal_executor, preventing a camera-search node from fighting cmd_vel or
causing a new room traversal.
"""
import json
import os
import time
import rospy
from std_msgs.msg import String


class RoomVisualSweepManager:
    def __init__(self):
        self.output = rospy.get_param("~output_dir", os.getcwd())
        self.path = os.path.join(self.output, "logs", "room_visual_sweep.jsonl")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.last_room = None
        rospy.Subscriber("/simenv/room_detection_context", String, self.context, queue_size=5)
        rospy.Subscriber("/simenv/local_rescan_result", String, self.result, queue_size=10)

    def log(self, value):
        try:
            with open(self.path, "a", encoding="utf-8") as f: f.write(json.dumps(value, sort_keys=True) + "\n")
        except OSError: pass

    def context(self, msg):
        try: c = json.loads(msg.data)
        except (ValueError, TypeError): return
        room = c.get("room_id") if c.get("enabled") else None
        if room and room != self.last_room:
            self.last_room = room
            self.log({"timestamp": time.time(), "event": "room_visual_search_armed", "room_id": room,
                      "directions_deg": [-90, -45, 0, 45, 90],
                      "policy": "door_anchored_first_interior_viewpoint_short_turn"})
        elif not room:
            self.last_room = None

    def result(self, msg):
        try: p = json.loads(msg.data)
        except (ValueError, TypeError): return
        if str(p.get("reason", "")).startswith("room_rgbd") or p.get("visual_sweep"):
            # The context may be cleared as soon as the planner accepts the
            # following EXIT goal.  Carry the originating room in the rescan
            # result so the audit cannot silently lose the completed sweep.
            room = p.get("room_id") or self.last_room
            self.log({"timestamp": time.time(), "event": "visual_sweep_complete",
                      "room_id": room, "result": p,
                      "directions_deg": [-90, -45, 0, 45, 90]})


if __name__ == "__main__":
    rospy.init_node("room_visual_sweep_manager")
    RoomVisualSweepManager(); rospy.spin()
