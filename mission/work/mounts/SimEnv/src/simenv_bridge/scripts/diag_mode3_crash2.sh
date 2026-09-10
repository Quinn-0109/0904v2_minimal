#!/bin/bash
# Trigger navi_mode=3 crash with the instrumented binary; capture what()+backtrace.
set -m
source /opt/ros/noetic/setup.bash
source /workspace/SimEnv/devel/setup.bash
source /workspace/SCAN-Planner/devel/setup.bash

rosnode kill /scan_planner_node /closed_loop_controller /scan_to_map /body_cmd_vel_driver /route_publisher 2>/dev/null
pkill -f 'simenv_scan.launch|scanplanner_simenv.launch|body_cmd_vel_driver|route_publisher' 2>/dev/null
sleep 2; yes | rosnode cleanup >/dev/null 2>&1

rosservice call /set_door_state main_entrance true 2>/dev/null
python3 /workspace/SimEnv/src/simenv_bridge/scripts/set_sim_physics.py 2>/dev/null &
nohup python3 /workspace/SimEnv/src/simenv_bridge/scripts/scan_to_map.py _input:=/livox/Pointcloud2 _output:=/registered_scan _target_frame:=map _min_range:=0.5 _max_range:=12.0 >/tmp/s2m.log 2>&1 &
nohup python3 /workspace/SimEnv/src/simenv_bridge/scripts/body_cmd_vel_driver.py _max_v:=1.5 _max_w:=1.0 _lock_z:=0.6 >/tmp/drv.log 2>&1 &
sleep 2
setsid roslaunch scan_planner simenv_scan.launch navi_mode:=3 >/tmp/sp3.log 2>&1 &
for i in $(seq 1 20); do rosnode list 2>/dev/null | grep -q /scan_planner_node && break; sleep 1; done
sleep 3
echo "scan_planner up: $(rosnode list 2>/dev/null | grep -c scan_planner)"
echo "publishing /initial_path (20 wp)..."
python3 /workspace/SimEnv/src/simenv_bridge/scripts/scanplanner_route_publisher.py /workspace/SimEnv/generated_building/layout_metadata.json >/tmp/rp.log 2>&1
sleep 15
pkill -f 'simenv_scan.launch' 2>/dev/null
rosnode kill /scan_planner_node /closed_loop_controller 2>/dev/null
echo "========= CAPTURED EXCEPTION ========="
grep -A40 -E "UNCAUGHT|CAUGHT|BACKTRACE" /tmp/sp3.log | head -50