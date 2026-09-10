#!/usr/bin/env python3
"""Provide an off-screen X display for Gazebo camera sensors.

Gazebo 11 disables DepthCameraSensor when DISPLAY is unset or inaccessible.
This guard owns Xvfb only when :99 is not already available, and leaves an
external display untouched.  It is infrastructure only: no navigation,
mapping, or detector data is changed here.
"""

import os
import subprocess
import time

import rospy


def _display_is_usable(display):
    """Return true only when an X server accepts connections on *display*."""
    try:
        check = subprocess.run(
            ["xdpyinfo", "-display", display],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=1.5,
            check=False)
        return check.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _remove_stale_display_files(number):
    """Remove only the proven-stale socket/lock belonging to this display."""
    for path in ("/tmp/.X11-unix/X{}".format(number),
                 "/tmp/.X{}-lock".format(number)):
        try:
            if os.path.lexists(path):
                os.unlink(path)
        except OSError as exc:
            raise RuntimeError("cannot remove stale X display file {}: {}".format(
                path, exc))


def main():
    rospy.init_node("xvfb_display_guard")
    display = rospy.get_param("~display", ":99")
    number = display.lstrip(":").split(".", 1)[0]
    socket_path = "/tmp/.X11-unix/X{}".format(number)
    process = None
    if not _display_is_usable(display):
        if os.path.lexists(socket_path):
            rospy.logwarn("Discarding stale, unreachable X display socket for %s", display)
            _remove_stale_display_files(number)
        process = subprocess.Popen(
            ["Xvfb", display, "-screen", "0", "1280x1024x24",
             "-nolisten", "tcp", "-ac"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Container cold-start / high CPU load can delay the first X socket
        # readiness well beyond eight seconds even though Xvfb has launched
        # successfully.  This is pre-mission infrastructure time, so wait
        # robustly instead of tearing down the whole ROS graph prematurely.
        startup_timeout = float(rospy.get_param("~startup_timeout_sec", 30.0))
        deadline = time.monotonic() + max(8.0, startup_timeout)
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("Xvfb exited before opening {}".format(display))
            if os.path.exists(socket_path) and _display_is_usable(display):
                break
            time.sleep(0.05)
        if not _display_is_usable(display):
            raise RuntimeError("Xvfb did not open {}".format(display))
        rospy.loginfo("Started Xvfb for Gazebo camera rendering on %s", display)
    else:
        rospy.loginfo("Reusing existing X display %s for Gazebo camera rendering", display)
    try:
        rospy.spin()
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    main()
