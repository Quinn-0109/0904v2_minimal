#!/usr/bin/env python3
"""Aggregate deployment checks in the same environment used by roslaunch.

Does not start Gazebo, a controller, or a ROS master. Dynamic-library probes
run in child processes so their dependencies cannot pollute later probes.
"""
import argparse
import glob
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--controller', required=True)
    args = parser.parse_args()
    failures = []
    checked = set()

    def check(label, action):
        try:
            detail = action()
            print('[runtime OK] {}{}'.format(label, ': ' + str(detail) if detail else ''), flush=True)
        except Exception as exc:
            failures.append(label)
            print('[runtime FAIL] {}: {}'.format(label, exc), flush=True)

    def command(argv):
        result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                universal_newlines=True, timeout=45)
        if result.returncode:
            raise RuntimeError('exit {}: {}'.format(result.returncode, result.stdout.strip()))
        return result.stdout.strip()

    def python_probe(code, *extra):
        return command([sys.executable, '-c', code] + list(extra))

    def library(path, load=False, ros_control_context=False):
        path = Path(path)
        with path.open('rb') as stream:
            if stream.read(4) != b'\x7fELF':
                raise RuntimeError('not an ELF file: ' + str(path))
        output = command(['ldd', str(path)])
        missing = [line.strip() for line in output.splitlines() if 'not found' in line]
        if missing:
            raise RuntimeError('; '.join(missing))
        if load:
            # The controller is loaded inside gazebo_ros_control after its
            # default RobotHWSim. These base libraries provide control_toolbox
            # PID symbols. Loading just the controller in an empty Python
            # process incorrectly rejects plugins which use those symbols.
            setup = ''
            if ros_control_context:
                setup = ("base=ctypes.CDLL('libgazebo_ros_control.so', mode=os.RTLD_NOW | os.RTLD_GLOBAL); "
                         "hw=ctypes.CDLL('libdefault_robot_hw_sim.so', mode=os.RTLD_NOW | os.RTLD_GLOBAL); ")
            python_probe('import ctypes, os, sys; ' + setup +
                         'ctypes.CDLL(sys.argv[1], mode=os.RTLD_NOW | os.RTLD_LOCAL)', str(path))
        return str(path)

    def executable(path):
        path = Path(path)
        if not os.access(str(path), os.X_OK):
            raise RuntimeError('not executable: ' + str(path))
        with path.open('rb') as stream:
            header = stream.readline(512)
        if header.startswith(b'\x7fELF'):
            library(path)
        elif header.startswith(b'#!'):
            if b'\r' in header:
                raise RuntimeError('CRLF shebang: ' + str(path))
            interpreter = shlex.split(header[2:].decode().strip())
            if not interpreter or not os.access(interpreter[0], os.X_OK):
                raise RuntimeError('missing shebang interpreter: ' + repr(interpreter))
            if Path(interpreter[0]).name == 'env' and len(interpreter) > 1:
                if interpreter[1] != '-S' and not shutil.which(interpreter[1]):
                    raise RuntimeError('interpreter not on PATH: ' + interpreter[1])
        return str(path)

    # Include service classes imported inside control_server.main(), not just
    # the top-level module which would miss genpy/generated-service failures.
    imports = [
        'import rospy, roslib, rospkg, roslaunch, genpy',
        'import cv2, numpy, yaml, tf, matplotlib',
        'from gazebo_msgs.srv import GetPhysicsProperties, SetPhysicsProperties, GetModelState, SetModelState',
        'from controller_manager_msgs.srv import ListControllers; from std_srvs.srv import Empty',
        'from sensor_msgs.msg import Image, CameraInfo, PointCloud2; from nav_msgs.msg import Odometry',
        'from unitree_legged_msgs.msg import LowCmd, LowState',
        'from building_generator_classic import control_server; print(control_server.__file__)',
        'from building_generator_interfaces.srv import CallElevator, CallElevatorResponse, SetDoorState, SetDoorStateResponse',
    ]
    print('[runtime] Python: ' + sys.executable, flush=True)
    for code in imports:
        check(code, lambda code=code: python_probe(code))
    check('validated RL executable and dependencies', lambda: executable(args.controller))

    def controller_plugin():
        exports = command(['rospack', 'plugins', '--attrib=plugin', 'controller_interface'])
        paths = []
        for line in exports.splitlines():
            fields = line.split(None, 1)
            if len(fields) == 2 and fields[0] == 'unitree_legged_control':
                paths.append(fields[1])
        if not paths:
            raise RuntimeError('unitree_legged_control plugin XML is not exported by rospack')
        for xml_path in paths:
            root = ET.parse(xml_path).getroot()
            libs = [root] if root.tag == 'library' else root.findall('.//library')
            for lib in libs:
                if not any(c.get('name') == 'unitree_legged_control/UnitreeJointController'
                           for c in lib.findall('class')):
                    continue
                relative = lib.attrib['path']
                if not relative.endswith('.so'):
                    relative += '.so'
                prefixes = [Path(p) for p in os.environ.get('CMAKE_PREFIX_PATH', '').split(':') if p]
                for prefix in prefixes:
                    candidate = prefix / relative
                    if (prefix / '.catkin').is_file() and candidate.is_file():
                        library(candidate, load=True, ros_control_context=True)
                        return '{} -> {}'.format(xml_path, candidate)
                raise RuntimeError('XML requires {}; absent from catkin CMAKE_PREFIX_PATH={}'.format(
                    relative, os.environ.get('CMAKE_PREFIX_PATH', '')))
        raise RuntimeError('UnitreeJointController class missing from exported XML')

    check('UnitreeJointController XML, prefix and dlopen with Gazebo ros_control context', controller_plugin)
    prefix = Path(os.environ['SIMENV_NATIVE_DEVEL_SPACE'])
    check('Livox Gazebo plugin and dependencies', lambda: library(prefix / 'lib/liblivox_laser_simulation.so', load=True))

    def launch_nodes(launch_name):
        import roslaunch
        import roslaunch.rlutil
        import roslib.packages
        path = roslaunch.rlutil.resolve_launch_arguments(['simenv_bridge', launch_name])[0]
        config = roslaunch.config.load_config_default([path], None)
        # Inspect expanded URDF rather than grepping xacro comments/disabled macros.
        descriptions = [param.value for key, param in config.params.items()
                        if key.endswith('/robot_description') and isinstance(param.value, str)]
        for description in descriptions:
            robot = ET.fromstring(description)
            for plugin in robot.findall('.//plugin'):
                filename = plugin.get('filename')
                if not filename or filename in checked:
                    continue
                checked.add(filename)

                def gazebo_plugin(filename=filename):
                    dirs = []
                    for variable in ('GAZEBO_PLUGIN_PATH', 'LD_LIBRARY_PATH'):
                        dirs.extend(p for p in os.environ.get(variable, '').split(':') if p)
                    dirs.extend(glob.glob('/usr/lib/*/gazebo-*/plugins'))
                    dirs.extend(['/opt/ros/noetic/lib', '/usr/lib/x86_64-linux-gnu'])
                    candidates = [Path(filename)] if os.path.isabs(filename) else [Path(d) / filename for d in dirs]
                    for candidate in candidates:
                        if candidate.is_file():
                            return library(candidate)
                    raise RuntimeError('expanded robot plugin not found: ' + filename)

                check('robot plugin ' + filename, gazebo_plugin)
        for node in config.nodes:
            label = node.package + '/' + node.type
            if label not in checked:
                checked.add(label)

                def node_check(node=node):
                    candidates = roslib.packages.find_node(node.package, node.type)
                    if not candidates:
                        raise RuntimeError('ROS node not found')
                    return executable(candidates[0])

                check(label, node_check)
            if node.launch_prefix:
                # Check the actual launch-prefix executable as well as node.type.
                first = shlex.split(node.launch_prefix)[0]
                resolved = shutil.which(first) if '/' not in first else first
                check(label + ' launch-prefix', lambda resolved=resolved: executable(resolved or 'MISSING_PREFIX'))
        return '{} nodes'.format(len(config.nodes))

    for name in ('scanplanner_three_floor_rl.launch', 'stair_descent_physical_smoke.launch'):
        check(name, lambda name=name: launch_nodes(name))
    if failures:
        print('[runtime] FAILED: {} check(s); simulation was not started.'.format(len(failures)))
        return 2
    print('[runtime] dependency checks passed; ROS/Gazebo live readiness still requires a run.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
