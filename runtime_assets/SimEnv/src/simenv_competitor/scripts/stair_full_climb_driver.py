#!/usr/bin/env python3
"""Physical acceptance test for both flights of the generated two-floor stair.

Debug-only: Gazebo model states are used to measure the actual climb.  This is
not part of the online F1 stair navigator; it proves that the policy and the
two-flight handoff can reach floor 1 before that navigator is integrated.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import rospy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool


def yaw_from_q(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def rpy_from_q(q):
    roll = math.atan2(2.0 * (q.w * q.x + q.y * q.z),
                      1.0 - 2.0 * (q.x * q.x + q.y * q.y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (q.w * q.y - q.z * q.x))))
    return roll, pitch


class FullStairTest:
    def __init__(self, args):
        self.a = args
        self.pose = None
        self.ready = False
        self.phase = "WAIT_READY"
        self.phase_start = None
        self.start_z = None
        self.rows = []
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=4)
        self.joy_pub = rospy.Publisher("/joy", Joy, queue_size=4)
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.models, queue_size=5)
        rospy.Subscriber("/locomotion_ready", Bool, self.ready_cb, queue_size=2)

    def models(self, msg):
        try:
            index = msg.name.index("a1_gazebo")
        except ValueError:
            return
        p = msg.pose[index]
        roll, pitch = rpy_from_q(p.orientation)
        self.pose = (p.position.x, p.position.y, p.position.z, yaw_from_q(p.orientation), roll, pitch)

    def ready_cb(self, msg):
        self.ready = self.ready or bool(msg.data)

    def hold_rl(self):
        joy = Joy()
        joy.header.stamp = rospy.Time.now()
        joy.axes = [0.0] * 8
        joy.buttons = [0] * 12
        joy.buttons[3] = 1
        self.joy_pub.publish(joy)

    def world_cmd(self, vx, vy):
        """World velocity -> body cmd_vel, using measured base yaw."""
        cmd = Twist()
        if self.pose:
            yaw = self.pose[3]
            cmd.linear.x = vx * math.cos(yaw) + vy * math.sin(yaw)
            cmd.linear.y = -vx * math.sin(yaw) + vy * math.cos(yaw)
        self.cmd_pub.publish(cmd)
        self.hold_rl()

    def turn_cmd(self, target_yaw):
        cmd = Twist()
        if self.pose:
            error = math.atan2(math.sin(target_yaw - self.pose[3]),
                               math.cos(target_yaw - self.pose[3]))
            cmd.angular.z = max(-self.a.turn_speed, min(self.a.turn_speed,
                                                         self.a.turn_gain * error))
        self.cmd_pub.publish(cmd)
        self.hold_rl()

    def world_turn_cmd(self, vx, vy, target_yaw):
        """Keep a world-frame translation while rotating on the landing."""
        cmd = Twist()
        if self.pose:
            yaw = self.pose[3]
            cmd.linear.x = vx * math.cos(yaw) + vy * math.sin(yaw)
            cmd.linear.y = -vx * math.sin(yaw) + vy * math.cos(yaw)
            error = math.atan2(math.sin(target_yaw - yaw),
                               math.cos(target_yaw - yaw))
            cmd.angular.z = max(-self.a.turn_speed, min(self.a.turn_speed,
                                                         self.a.turn_gain * error))
        self.cmd_pub.publish(cmd)
        self.hold_rl()

    def turn_to(self, target_yaw, phase, rate):
        self.change(phase)
        while not rospy.is_shutdown():
            if self.unsafe(): return False, "tipped_" + phase.lower()
            if self.expired(): return False, phase.lower() + "_timeout"
            error = math.atan2(math.sin(target_yaw - self.pose[3]),
                               math.cos(target_yaw - self.pose[3]))
            if abs(error) <= self.a.turn_tolerance:
                self.cmd_pub.publish(Twist()); self.hold_rl()
                return True, ""
            self.turn_cmd(target_yaw); self.record(); rate.sleep()
        return False, "ros_shutdown"

    def moving_turn_to(self, target_yaw, phase, rate):
        """Turn while translating across the landing; policy is not stable at rest.

        The intermediate landing is too short for an RL-policy restart or a
        static 180-degree pivot.  Retain the stair policy and use a small
        world-frame cross-landing translation.  Keeping the translation in
        world coordinates prevents the arc from curving back down flight A as
        the base yaw changes.
        """
        self.change(phase)
        while not rospy.is_shutdown():
            if self.unsafe(): return False, "tipped_" + phase.lower()
            if self.expired(): return False, phase.lower() + "_timeout"
            error = math.atan2(math.sin(target_yaw - self.pose[3]),
                               math.cos(target_yaw - self.pose[3]))
            # Complete both heading and lateral alignment before beginning the
            # next flight.  Reaching the heading alone leaves the robot on the
            # outside edge of flight B, where the first tread causes a tip.
            if (abs(error) <= self.a.turn_tolerance and
                    self.pose[0] >= self.a.flight_b_entry_x):
                return True, ""
            self.world_turn_cmd(self.a.moving_turn_world_x_speed,
                                self.a.moving_turn_world_y_speed,
                                target_yaw)
            self.record(); rate.sleep()
        return False, "ros_shutdown"

    def unsafe(self):
        return self.pose is not None and (abs(self.pose[4]) > self.a.tip_rad or
                                          abs(self.pose[5]) > self.a.tip_rad)

    def expired(self):
        return (rospy.Time.now() - self.phase_start).to_sec() > self.a.phase_timeout

    def change(self, name):
        self.phase = name
        self.phase_start = rospy.Time.now()
        rospy.loginfo("[FULL-STAIR] %s", name)

    def record(self):
        if self.pose:
            self.rows.append({"t_sim": rospy.Time.now().to_sec(), "phase": self.phase,
                              "x": self.pose[0], "y": self.pose[1], "z": self.pose[2],
                              "yaw": self.pose[3], "roll": self.pose[4], "pitch": self.pose[5]})

    def run(self):
        rate = rospy.Rate(50)
        deadline = rospy.Time.now() + rospy.Duration(self.a.ready_timeout)
        while not rospy.is_shutdown() and not self.ready:
            if rospy.Time.now() > deadline:
                return self.finish(False, "locomotion_ready_timeout")
            # Keep the RL request alive even during the controller's blend /
            # stationary-hold window.  Otherwise its own unstable-takeover
            # protection can select Passive before readiness is latched.
            self.hold_rl()
            rate.sleep()
        # A brief, recorded stationary baseline after controller readiness.
        self.change("BASELINE")
        while (rospy.Time.now() - self.phase_start).to_sec() < 1.0:
            self.cmd_pub.publish(Twist()); self.hold_rl(); self.record(); rate.sleep()
        self.start_z = self.pose[2]
        if self.a.flight_b_only:
            self.change("FLIGHT_B")
            while not rospy.is_shutdown():
                if self.unsafe(): return self.finish(False, "tipped_flight_b")
                if self.expired(): return self.finish(False, "flight_b_timeout")
                if self.pose[2] - self.start_z >= self.a.flight_b_rise:
                    return self.finish(True, "second_flight_height_reached")
                self.world_cmd(0.0, -self.a.speed); self.record(); rate.sleep()
            return self.finish(False, "ros_shutdown")
        self.change("FLIGHT_A")
        while not rospy.is_shutdown():
            if self.unsafe(): return self.finish(False, "tipped_flight_a")
            if self.expired(): return self.finish(False, "flight_a_timeout")
            rise = self.pose[2] - self.start_z
            # Height alone is not sufficient: the base can exceed 1.05 m on
            # the final few treads while it is still pitched on the flight.
            # Begin the hairpin only after it has reached the flat landing.
            if rise >= self.a.flight_a_rise and self.pose[1] >= self.a.flight_a_end_y:
                self.change("MID_LANDING_SETTLE")
                break
            self.world_cmd(0.0, self.a.speed); self.record(); rate.sleep()
        # Do not send zero velocity on the landing: with the stair policy that
        # triggers a passive fallback before a static turn can complete.
        ok, reason = self.moving_turn_to(self.a.flight_b_heading,
                                         "MID_LANDING_MOVING_TURN", rate)
        if not ok: return self.finish(False, reason)
        self.change("FLIGHT_B")
        while not rospy.is_shutdown():
            if self.unsafe(): return self.finish(False, "tipped_flight_b")
            if self.expired(): return self.finish(False, "flight_b_timeout")
            rise = self.pose[2] - self.start_z
            if rise >= self.a.total_rise:
                return self.finish(True, "second_floor_height_reached")
            self.world_cmd(0.0, -self.a.speed); self.record(); rate.sleep()
        return self.finish(False, "ros_shutdown")

    def finish(self, success, reason):
        self.cmd_pub.publish(Twist())
        self.hold_rl()
        out = Path(self.a.output_dir); out.mkdir(parents=True, exist_ok=True)
        with (out / "full_stair_track.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("t_sim", "phase", "x", "y", "z", "yaw", "roll", "pitch"))
            writer.writeheader(); writer.writerows(self.rows)
        elapsed = self.rows[-1]["t_sim"] - self.rows[0]["t_sim"] if self.rows else 0.0
        max_z = max((row["z"] for row in self.rows), default=0.0)
        # Preserve the time spent in every completed/attempted phase.  This is
        # deliberately measured from simulation time rather than wall time so
        # Gazebo start-up and rendering do not contaminate the stair metric.
        phase_durations = {}
        if self.rows:
            names = [row["phase"] for row in self.rows]
            for name in dict.fromkeys(names):
                phase_rows = [row for row in self.rows if row["phase"] == name]
                phase_durations[name] = round(
                    float(phase_rows[-1]["t_sim"]) - float(phase_rows[0]["t_sim"]), 3)
        summary = {"success": success, "reason": reason, "elapsed_from_ready_s": round(elapsed, 3),
                   "start_z_m": round(self.start_z or 0.0, 3), "max_z_m": round(max_z, 3),
                   "height_gain_m": round(max_z - (self.start_z or 0.0), 3), "rows": len(self.rows),
                   "phase_durations_s": phase_durations}
        (out / "full_stair_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        self._write_plot(out, summary)
        rospy.loginfo("[FULL-STAIR] %s", summary)
        return 0 if success else 1

    def _write_plot(self, out, summary):
        """Write an immediately readable two-flight debug visualization."""
        if not self.rows:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as error:
            rospy.logwarn("[FULL-STAIR] plot unavailable: %s", error)
            return
        colors = {"BASELINE": "#808080", "FLIGHT_A": "#1976d2",
                  "MID_LANDING_MOVING_TURN": "#f57c00", "FLIGHT_B": "#2e7d32"}
        fig, (plan, elevation) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        for phase in dict.fromkeys(row["phase"] for row in self.rows):
            rows = [row for row in self.rows if row["phase"] == phase]
            x = [row["x"] for row in rows]
            y = [row["y"] for row in rows]
            t = [row["t_sim"] - self.rows[0]["t_sim"] for row in rows]
            z = [row["z"] for row in rows]
            label = phase.replace("_", " ").title()
            plan.plot(x, y, color=colors.get(phase, "#444444"), linewidth=2.5, label=label)
            elevation.plot(t, z, color=colors.get(phase, "#444444"), linewidth=2.5, label=label)
        plan.scatter([self.rows[0]["x"]], [self.rows[0]["y"]], c="#00a000", s=70, marker="o", label="F1 stair entry")
        plan.scatter([self.rows[-1]["x"]], [self.rows[-1]["y"]], c="#111111", s=70, marker="X", label="end")
        plan.set(title="Two-flight stair trajectory", xlabel="world x (m)", ylabel="world y (m)")
        plan.axis("equal"); plan.grid(alpha=.25); plan.legend(fontsize=8)
        elevation.set(title="Height reached: %.3f m (%s)" % (summary["max_z_m"], "PASS" if summary["success"] else "FAIL"),
                      xlabel="elapsed after RL ready (s)", ylabel="base z (m)")
        elevation.grid(alpha=.25); elevation.legend(fontsize=8)
        fig.savefig(str(out / "full_stair_trajectory.png"), dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--speed", type=float, default=0.8)
    parser.add_argument("--landing-speed", type=float, default=0.55)
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    parser.add_argument("--ready-timeout", type=float, default=90.0)
    parser.add_argument("--flight-a-rise", type=float, default=1.05)
    parser.add_argument("--flight-a-end-y", type=float, default=4.40,
                        help="world-y landing threshold before the hairpin")
    parser.add_argument("--total-rise", type=float, default=2.20)
    parser.add_argument("--flight-b-only", action="store_true",
                        help="validate the second flight from its known landing centreline")
    parser.add_argument("--flight-b-rise", type=float, default=1.05)
    parser.add_argument("--landing-x-margin", type=float, default=0.0)
    parser.add_argument("--phase-timeout", type=float, default=25.0)
    parser.add_argument("--tip-rad", type=float, default=0.6)
    parser.add_argument("--turn-speed", type=float, default=0.35)
    parser.add_argument("--turn-gain", type=float, default=0.7)
    parser.add_argument("--turn-tolerance", type=float, default=0.15)
    parser.add_argument("--moving-turn-world-x-speed", type=float, default=0.40,
                        help="world-frame cross-landing speed towards flight B")
    parser.add_argument("--moving-turn-world-y-speed", type=float, default=0.0)
    parser.add_argument("--flight-b-entry-x", type=float, default=-2.68,
                        help="world-x centreline target before starting flight B")
    parser.add_argument("--flight-b-heading", type=float, default=-1.20,
                        help="acceptable base yaw before world-frame flight-B climb")
    args = parser.parse_args()
    rospy.init_node("stair_full_climb_driver")
    return FullStairTest(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
