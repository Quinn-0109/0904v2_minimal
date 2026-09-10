#!/bin/bash
# Diagnose navi_mode=3 boost::lock_error via gdb catch-throw. Runs in simenv-run.
set -m
source /opt/ros/noetic/setup.bash
source /workspace/SimEnv/devel/setup.bash
source /workspace/SCAN-Planner/devel/setup.bash

rosnode kill /scan_planner_node /closed_loop_controller /scan_to_map /body_cmd_vel_driver /route_publisher 2>/dev/null
pkill -f 'simenv_scan.launch|scanplanner_simenv.launch|body_cmd_vel_driver|route_publisher' 2>/dev/null
sleep 2; yes | rosnode cleanup >/dev/null 2>&1

# reset A1
python3 - <<'PY'
import math, rospy
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
rospy.init_node('rz', anonymous=True)
rospy.wait_for_service('/gazebo/set_model_state', timeout=10)
st=ModelState(); st.model_name='a1_gazebo'; st.reference_frame='world'
st.pose.position.y=2.0; st.pose.position.z=0.6
st.pose.orientation.z=math.sin(1.5708/2); st.pose.orientation.w=math.cos(1.5708/2)
rospy.ServiceProxy('/gazebo/set_model_state',SetModelState)(st)
PY
rosservice call /set_door_state main_entrance true 2>/dev/null
python3 /workspace/SimEnv/src/simenv_bridge/scripts/set_sim_physics.py 2>/dev/null &
nohup python3 /workspace/SimEnv/src/simenv_bridge/scripts/scan_to_map.py _input:=/livox/Pointcloud2 _output:=/registered_scan _target_frame:=map _min_range:=0.5 _max_range:=12.0 >/tmp/s2m.log 2>&1 &
nohup python3 /workspace/SimEnv/src/simenv_bridge/scripts/body_cmd_vel_driver.py _max_v:=1.5 _max_w:=1.0 _lock_z:=0.6 >/tmp/drv.log 2>&1 &
sleep 2
setsid roslaunch scan_planner simenv_scan.launch navi_mode:=3 >/tmp/sp3.log 2>&1 &
for i in $(seq 1 20); do rosnode list 2>/dev/null | grep -q /scan_planner_node && break; sleep 1; done
sleep 2
PID=$(pgrep -f 'devel/lib/scan_planner/scan_planner_node' | head -1)
echo "scan_planner_node PID=$PID"
[ -z "$PID" ] && { echo "node not up"; tail -5 /tmp/sp3.log; exit 1; }

echo "attaching gdb (catch throw)..."
timeout 40 gdb -p "$PID" -batch \
  -ex 'set pagination off' -ex 'set print thread-events off' \
  -ex 'catch throw' -ex 'continue' \
  -ex 'bt 50' -ex 'info args' -ex 'thread apply all bt 20' \
  > /tmp/bt.txt 2>&1 &
GDB=$!
sleep 4
echo "publishing /initial_path (20 waypoints)..."
python3 /workspace/SimEnv/src/simenv_bridge/scripts/scanplanner_route_publisher.py /workspace/SimEnv/generated_building/layout_metadata.json >/tmp/rp.log 2>&1 &
sleep 30
kill $GDB 2>/dev/null
pkill -f 'simenv_scan.launch' 2>/dev/null
rosnode kill /scan_planner_node /closed_loop_controller 2>/dev/null
echo "=== /tmp/bt.txt (catch-throw backtrace) ==="
cat /tmp/bt.txt | grep -vE "^\[Thread|^\[New|^\[Detaching|Reading symbols|^warning: |^Using |^0x|^\$|^__|^\(gdb\)" | head -70
echo "=== route_publisher log ==="; tail -3 /tmp/rp.log