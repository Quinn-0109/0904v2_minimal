#!/usr/bin/env python3
"""Minimal keyboard teleop for the non-RL A1 trotting controller.

The node publishes the Unitree controller's ROS control topic (``/joy``).
Button 4 selects the classical ``State_Trotting`` controller; it does not
select either of the RL states.  Motion commands persist until another key or
space is pressed, matching the usual ROS keyboard teleop behaviour.
"""

import os
import select
import sys
import termios
import time
import tty

import rospy
from sensor_msgs.msg import Joy


BUTTON_FIXED_STAND = 1
BUTTON_CLASSIC_TROTTING = 4
BUTTON_RESET = 10


def axes_for_key(key, linear_scale=0.35, angular_scale=0.45):
    """Return Unitree Joy axes for a motion key, or ``None`` if unhandled."""
    axes = [0.0] * 6
    key = key.lower()
    if key == "w":
        axes[1] = float(linear_scale)
    elif key == "s":
        axes[1] = -float(linear_scale)
    elif key == "a":
        # State_Trotting applies yaw=-rx: negative rx is a left turn.
        axes[3] = -float(angular_scale)
    elif key == "d":
        axes[3] = float(angular_scale)
    elif key in (" ", "\n", "\r"):
        pass
    else:
        return None
    return axes


class ManualSlamTeleop:
    def __init__(self):
        self._linear_scale = float(rospy.get_param("~linear_scale", 0.35))
        self._angular_scale = float(rospy.get_param("~angular_scale", 0.45))
        self._repeat_rate = max(5.0, float(rospy.get_param("~repeat_rate", 20.0)))
        self._input_device = str(rospy.get_param("~input_device", "/dev/tty"))
        self._stand_seconds = max(
            0.0, float(rospy.get_param("~stand_seconds", 5.0))
        )
        self._axes = [0.0] * 6
        self._publisher = rospy.Publisher("/joy", Joy, queue_size=3)
        self._stream = None
        self._settings = None

    def _message(self, stand=False, reset=False):
        message = Joy()
        message.header.stamp = rospy.Time.now()
        message.axes = list(self._axes)
        message.buttons = [0] * 11
        if reset:
            message.buttons[BUTTON_RESET] = 1
        else:
            message.buttons[
                BUTTON_FIXED_STAND if stand else BUTTON_CLASSIC_TROTTING
            ] = 1
        return message

    def _publish(self, stand=False, reset=False):
        self._publisher.publish(self._message(stand=stand, reset=reset))

    def _open_terminal(self):
        candidates = [self._input_device]
        # roslaunch intentionally gives children /dev/null on stdin and they
        # therefore have no /dev/tty.  The launch parent still owns the IDE or
        # SSH pseudo terminal; discover it through the process ancestry.
        pid = os.getppid()
        for _ in range(6):
            for fd in (0, 1, 2):
                try:
                    target = os.path.realpath("/proc/{}/fd/{}".format(pid, fd))
                except OSError:
                    continue
                if target.startswith("/dev/pts/") or target.startswith("/dev/tty"):
                    candidates.append(target)
            try:
                with open("/proc/{}/stat".format(pid), encoding="utf-8") as stream:
                    fields = stream.read().split()
                pid = int(fields[3])
            except (OSError, ValueError, IndexError):
                break

        last_error = None
        for candidate in dict.fromkeys(candidates):
            try:
                stream = open(candidate, "rb", buffering=0)
                settings = termios.tcgetattr(stream.fileno())
                tty.setcbreak(stream.fileno())
                self._stream = stream
                self._settings = settings
                rospy.loginfo("Keyboard input attached to %s", candidate)
                return True
            except (OSError, termios.error) as error:
                last_error = error
        rospy.logerr(
            "No interactive TTY found (%s). SLAM will keep running with the "
            "robot stopped; start this node from an interactive terminal to "
            "enable WASD.",
            last_error,
        )
        return False

    def _restore_terminal(self):
        if self._stream is not None and self._settings is not None:
            termios.tcsetattr(
                self._stream.fileno(), termios.TCSADRAIN, self._settings
            )
        if self._stream is not None:
            self._stream.close()

    def run(self):
        interactive = self._open_terminal()
        print("\n=== FAST-LIO2 manual mapping teleop (classic Trotting, no RL) ===")
        print("W: forward  S: backward  A: turn left  D: turn right")
        print("Space: stop  R: reset a fallen robot  Q: stop and quit teleop\n")
        period = 1.0 / self._repeat_rate
        next_publish = time.monotonic()
        started = time.monotonic()
        try:
            while not rospy.is_shutdown():
                if not interactive:
                    self._publish()
                    time.sleep(period)
                    continue
                timeout = max(0.0, next_publish - time.monotonic())
                readable, _, _ = select.select([self._stream], [], [], timeout)
                if readable:
                    raw = os.read(self._stream.fileno(), 1)
                    if not raw:
                        continue
                    key = raw.decode("utf-8", errors="ignore")
                    if key.lower() == "q" or key == "\x03":
                        self._axes = [0.0] * 6
                        self._publish()
                        break
                    if key.lower() == "r":
                        self._axes = [0.0] * 6
                        # Hold RESET long enough for the 500 Hz FSM to latch it,
                        # then restart the fixed-stand warmup sequence.
                        for _ in range(5):
                            self._publish(reset=True)
                            time.sleep(0.03)
                        started = time.monotonic()
                        rospy.loginfo("Manual teleop command: reset_robot")
                        continue
                    axes = axes_for_key(
                        key, self._linear_scale, self._angular_scale
                    )
                    if axes is not None:
                        self._axes = axes
                        label = {
                            "w": "forward",
                            "s": "backward",
                            "a": "turn_left",
                            "d": "turn_right",
                            " ": "stop",
                        }.get(key.lower(), "stop")
                        rospy.loginfo("Manual teleop command: %s", label)
                if time.monotonic() >= next_publish:
                    # Repeating START makes startup robust if the controller
                    # subscribes after Gazebo has finished spawning.  In the
                    # Trotting state START is harmless and never selects RL.
                    self._publish(
                        stand=time.monotonic() - started < self._stand_seconds
                    )
                    next_publish = time.monotonic() + period
        finally:
            self._axes = [0.0] * 6
            for _ in range(3):
                self._publish(stand=True)
                time.sleep(0.03)
            self._restore_terminal()


if __name__ == "__main__":
    rospy.init_node("manual_slam_teleop")
    ManualSlamTeleop().run()
