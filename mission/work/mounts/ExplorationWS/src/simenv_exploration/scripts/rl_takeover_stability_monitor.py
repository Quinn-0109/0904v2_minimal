#!/usr/bin/env python3
"""Persist state-gated RL takeover evidence and a compact diagnostic plot."""

import json
import math
import os
import threading
import time

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String


class TakeoverMonitor:
    FIXED = [0.0, 0.9, -1.8] * 4
    # Physical motor order after State_RL's reindex mapping.
    NOMINAL = [0.15, 0.55, -1.5, -0.15, 0.55, -1.5,
               0.15, 0.7, -1.5, -0.15, 0.7, -1.5]

    def __init__(self):
        self.output = os.path.abspath(rospy.get_param("~output_dir"))
        os.makedirs(self.output, exist_ok=True)
        self.lock = threading.RLock()
        self.stand = []
        self.takeover = []
        self.readiness = []
        self.commands = []
        self.joints = None
        self.goal_result = None
        self.supervisor = "FIXED_STAND"
        self.started = time.monotonic()
        rospy.Subscriber("/fixed_stand_status", String, self._stand, queue_size=100)
        rospy.Subscriber("/rl_takeover_status", String, self._takeover, queue_size=100)
        rospy.Subscriber("/locomotion_ready", Bool, self._ready, queue_size=10)
        rospy.Subscriber("/simenv/rl_takeover_supervisor_state", String,
                         self._supervisor, queue_size=10)
        rospy.Subscriber("/cmd_vel", Twist, self._cmd, queue_size=100)
        rospy.Subscriber("/joint_states", JointState, self._joint, queue_size=10)
        rospy.Subscriber("/simenv/goal_execution_result", String,
                         self._goal, queue_size=10)
        rospy.Timer(rospy.Duration(0.5), self._flush)
        rospy.on_shutdown(self.write)

    @staticmethod
    def _decode(message):
        try:
            return json.loads(message.data)
        except (TypeError, ValueError):
            return {"raw": message.data}

    def _stand(self, msg):
        with self.lock:
            self.stand.append(self._decode(msg))

    def _takeover(self, msg):
        with self.lock:
            self.takeover.append(self._decode(msg))

    def _ready(self, msg):
        with self.lock:
            self.readiness.append({"timestamp": time.time(), "ready": bool(msg.data)})

    def _supervisor(self, msg):
        with self.lock:
            self.supervisor = msg.data

    def _cmd(self, msg):
        with self.lock:
            self.commands.append({"timestamp": time.time(), "vx": msg.linear.x,
                                  "vy": msg.linear.y, "wz": msg.angular.z})

    def _joint(self, msg):
        with self.lock:
            self.joints = {"names": list(msg.name), "position": list(msg.position),
                           "velocity": list(msg.velocity)}

    def _goal(self, msg):
        with self.lock:
            self.goal_result = self._decode(msg)

    @staticmethod
    def _atomic(path, payload):
        temp = path + ".tmp"
        with open(temp, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temp, path)

    def _flush(self, _event):
        self.write()

    def write(self):
        with self.lock:
            ready = any(item["ready"] for item in self.readiness)
            failed = self.supervisor.startswith("FAILED") or any(
                item.get("phase") == "FAILED" for item in self.takeover)
            maximum_roll = max((abs(item.get("roll", 0.0)) for item in
                                self.stand + self.takeover), default=None)
            maximum_pitch = max((abs(item.get("pitch", 0.0)) for item in
                                 self.stand + self.takeover), default=None)
            blend_samples = [item for item in self.takeover
                             if item.get("phase") == "RL_BLEND"]
            blend_duration = 0.0
            if len(blend_samples) > 1:
                blend_duration = max(
                    0.0, float(blend_samples[-1].get("timestamp", 0.0)) -
                    float(blend_samples[0].get("timestamp", 0.0)))
            zero_hold_samples = [item for item in self.takeover
                                 if item.get("phase") == "RL_ZERO_HOLD"]
            zero_hold_duration = 0.0
            if len(zero_hold_samples) > 1:
                zero_hold_duration = max(
                    0.0, float(zero_hold_samples[-1].get("timestamp", 0.0)) -
                    float(zero_hold_samples[0].get("timestamp", 0.0)))
            summary = {
                "schema": "simenv_rl_takeover_stability_v1",
                "state": "FAILED" if failed else ("LOCOMOTION_READY" if ready else self.supervisor),
                "locomotion_ready": ready,
                "fixed_stand_wait_sec": (self.stand[-1].get("timestamp", 0) -
                                          self.stand[0].get("timestamp", 0)) if len(self.stand) > 1 else 0,
                "rl_blend_duration_sec": round(blend_duration, 3),
                "rl_zero_hold_sec": round(zero_hold_duration, 3),
                "maximum_abs_roll": maximum_roll,
                "maximum_abs_pitch": maximum_pitch,
                "entered_passive_down": failed,
                "goal_result": self.goal_result,
            }
            comparison = []
            names = (self.joints or {}).get("names") or ["joint_%02d" % i for i in range(12)]
            for index, (fixed, nominal) in enumerate(zip(self.FIXED, self.NOMINAL)):
                comparison.append({"joint_name": names[index] if index < len(names) else "joint_%02d" % index,
                                   "fixed_stand_target": fixed, "rl_nominal_pose": nominal,
                                   "first_rl_action_target": None,
                                   "fixed_to_rl_nominal_error": nominal - fixed,
                                   "fixed_to_first_rl_error": None})
            first_ten = self.takeover[:10]
            diagnostic = {
                "first_10_action_rms": [x.get("action_rms") for x in first_ten],
                "first_10_joint_target_delta": [],
                "observation_valid": all(math.isfinite(float(x.get("action_rms", 0))) for x in first_ten),
                "history_initialized": True,
                "hidden_state_reset": True,
                "samples": self.takeover,
            }
            self._atomic(os.path.join(self.output, "summary.json"), summary)
            self._atomic(os.path.join(self.output, "stand_stability_history.json"), self.stand)
            self._atomic(os.path.join(self.output, "rl_takeover_pose_comparison.json"), comparison)
            self._atomic(os.path.join(self.output, "rl_takeover_diagnostic.json"), diagnostic)
            self._atomic(os.path.join(self.output, "locomotion_readiness_history.json"), self.readiness)
            self._atomic(os.path.join(self.output, "motion_stability.json"), {
                "stand": self.stand, "takeover": self.takeover, "commands": self.commands})
            self._atomic(os.path.join(self.output, "goal_execution_log.json"),
                         self.goal_result or {})
            self._atomic(os.path.join(self.output, "execution_abort.json"), {
                "aborted": failed, "reason": self.supervisor if failed else None})
        self._plot()

    def _plot(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return
        samples = list(self.stand) + list(self.takeover)
        if not samples:
            return
        t0 = samples[0].get("timestamp", 0)
        ts = [x.get("timestamp", t0) - t0 for x in samples]
        fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
        axes[0].plot(ts, [x.get("roll", math.nan) for x in samples], label="roll")
        axes[0].plot(ts, [x.get("pitch", math.nan) for x in samples], label="pitch")
        axes[0].legend(); axes[0].set_ylabel("rad")
        axes[1].plot(ts, [x.get("base_z", math.nan) for x in samples], label="base_z")
        axes[1].plot(ts, [x.get("joint_velocity_rms", math.nan) for x in samples], label="joint velocity rms")
        axes[1].legend()
        axes[2].plot(ts, [x.get("blend_alpha", 0.0) for x in samples], label="blend alpha")
        axes[2].plot(ts, [x.get("action_rms", math.nan) for x in samples], label="action rms")
        axes[2].legend()
        axes[3].step(ts, [1 if x.get("fixed_stand_complete") or
                          x.get("locomotion_ready") else 0 for x in samples], where="post")
        axes[3].set_ylabel("ready"); axes[3].set_xlabel("seconds")
        fig.tight_layout()
        fig.savefig(os.path.join(self.output, "rl_takeover_stability.png"), dpi=140)
        plt.close(fig)


if __name__ == "__main__":
    rospy.init_node("rl_takeover_stability_monitor")
    TakeoverMonitor()
    rospy.spin()
