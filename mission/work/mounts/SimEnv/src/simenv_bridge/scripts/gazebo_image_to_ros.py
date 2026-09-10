#!/usr/bin/env python3
"""Build and execute the tiny Gazebo-transport to ROS image bridge.

The archived robot URDF creates the native Gazebo RGB camera but its ROS
sensor plugin is dropped during URDF-to-SDF conversion.  Building this helper
in /tmp keeps generated artifacts out of the read-only source overlay while
preserving the real rendered pixels used by online red-ball detection.
"""

import hashlib
import os
from pathlib import Path
import shlex
import subprocess
import sys


def source_path():
    local = Path(__file__).resolve().with_suffix(".cpp")
    if local.is_file():
        return local
    import rospkg
    return Path(rospkg.RosPack().get_path("simenv_bridge")) / "scripts" / local.name


def build_binary(source):
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    directory = Path("/tmp/simenv_bridge_runtime")
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / ("gazebo_image_to_ros_" + digest)
    if target.is_file() and os.access(str(target), os.X_OK):
        return target
    flags = shlex.split(subprocess.check_output(
        ["pkg-config", "--cflags", "--libs", "gazebo", "roscpp", "sensor_msgs", "std_msgs"],
        text=True,
    ))
    temporary = target.with_name(target.name + ".{}.tmp".format(os.getpid()))
    command = ["g++", "-O2", "-std=c++17", str(source), "-o", str(temporary)] + flags
    subprocess.run(command, check=True)
    os.chmod(str(temporary), 0o755)
    os.replace(str(temporary), str(target))
    return target


def main():
    source = source_path()
    if not source.is_file():
        raise SystemExit("Gazebo image bridge source missing: {}".format(source))
    binary = build_binary(source)
    os.execv(str(binary), [str(binary)] + sys.argv[1:])


if __name__ == "__main__":
    main()
