#!/bin/bash
# Run the four-room SCAN-Planner mission inside simenv-run.
source /opt/ros/noetic/setup.bash
source /workspace/SimEnv/devel/setup.bash
source /workspace/SCAN-Planner/devel/setup.bash
set -u
set -o pipefail

MAX_DURATION=200
RESULTS=/workspace/SimEnv/results/scan_planner_demo
RUN_LOG="$RESULTS/scanplanner_simenv_fixed.log"
RECORD="$RESULTS/scanplanner_simenv_fixed_record.npz"
DETECTED="$RESULTS/detected_danger.json"
EVALUATION="$RESULTS/evaluation_result.json"
mkdir -p "$RESULTS"
rm -f "$RECORD" "$DETECTED" "$EVALUATION" "$RESULTS/scanplanner_camera_fixed.mp4"

stop_pipeline() {
  if [ -n "${LP:-}" ]; then
    kill -- -"$LP" 2>/dev/null || true
  fi
  rosnode kill /scan_planner_node /closed_loop_controller /body_cmd_vel_driver \
    /scan_to_map /scanplanner_goal_sequencer /danger_detector \
    /camera_video_recorder 2>/dev/null || true
}
trap stop_pipeline EXIT INT TERM

rosnode kill /scan_planner_node /closed_loop_controller /scan_to_map \
  /scanplanner_goal_sequencer /danger_detector /camera_video_recorder 2>/dev/null || true
pkill -f body_cmd_vel_driver.py 2>/dev/null || true
pkill -f scanplanner_simenv.launch 2>/dev/null || true
sleep 2
yes | rosnode cleanup >/dev/null 2>&1 || true

check_topic() {
  local topic=$1
  if ! timeout 10 rostopic echo -n1 --noarr "$topic" >/dev/null 2>&1; then
    echo "ERROR: required topic unavailable: $topic" | tee -a "$RUN_LOG"
    exit 2
  fi
  echo "topic OK: $topic" | tee -a "$RUN_LOG"
}

: > "$RUN_LOG"
check_topic /Odometry_gazebo
check_topic /livox/Pointcloud2
check_topic /real_sense/rgb/image_raw
check_topic /real_sense/depth/image_raw

WORLD_PROPERTIES=$(rosservice call /gazebo/get_world_properties 2>/dev/null || true)
if ! grep -q "generated_building" <<<"$WORLD_PROPERTIES"; then
  echo "ERROR: generated_building is missing from Gazebo" | tee -a "$RUN_LOG"
  exit 2
fi
if ! grep -q "danger_red_sphere" <<<"$WORLD_PROPERTIES"; then
  echo "ERROR: danger_red_sphere models are missing from Gazebo" | tee -a "$RUN_LOG"
  exit 2
fi
echo "world OK: generated_building and danger_red_sphere models present" | tee -a "$RUN_LOG"

python3 - <<'PY' | tee -a "$RUN_LOG"
import math
import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState

rospy.init_node("reset_a1", anonymous=True)
rospy.wait_for_service("/gazebo/set_model_state", timeout=10)
state = ModelState()
state.model_name = "a1_gazebo"
state.reference_frame = "world"
state.pose.position.x = 0.0
state.pose.position.y = 2.0
state.pose.position.z = 0.6
state.pose.orientation.z = math.sin(math.pi / 4.0)
state.pose.orientation.w = math.cos(math.pi / 4.0)
response = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)(state)
print("reset A1 success=%s" % response.success)
PY
sleep 0.2
rosservice call /set_door_state main_entrance true >/dev/null 2>&1 || true

setsid roslaunch simenv_bridge scanplanner_simenv.launch \
  navi_mode:=1 results_dir:="$RESULTS" >> "$RUN_LOG" 2>&1 &
LP=$!
for _ in $(seq 1 30); do
  if rosnode list 2>/dev/null | grep -qx /scan_planner_node && \
     rosnode list 2>/dev/null | grep -qx /danger_detector && \
     rosnode list 2>/dev/null | grep -qx /camera_video_recorder; then
    break
  fi
  sleep 1
done

LOADED_MAX_VEL=$(rosparam get /scan_planner_node/manager/max_vel 2>/dev/null || true)
LOADED_MAX_ACC=$(rosparam get /scan_planner_node/manager/max_acc 2>/dev/null || true)
echo "loaded manager/max_vel=$LOADED_MAX_VEL" | tee -a "$RUN_LOG"
echo "loaded manager/max_acc=$LOADED_MAX_ACC" | tee -a "$RUN_LOG"
if [ "$LOADED_MAX_VEL" != "1.5" ]; then
  echo "ERROR: expected runtime max_vel=1.5" | tee -a "$RUN_LOG"
  exit 3
fi
if [ "$LOADED_MAX_ACC" != "2.0" ]; then
  echo "ERROR: expected runtime max_acc=2.0" | tee -a "$RUN_LOG"
  exit 3
fi

python3 /workspace/SimEnv/src/simenv_bridge/scripts/scanplanner_record.py \
  _duration:="$MAX_DURATION" _out:="$RECORD" >> "$RUN_LOG" 2>&1

python3 - "$RECORD" <<'PY' | tee -a "$RUN_LOG"
import sys
import numpy as np

record = np.load(sys.argv[1], allow_pickle=True)
speed = np.asarray(record["cmd_speed"], dtype=float)
moving = speed[speed > 0.05]
print("applied_command_speed mean=%.3f p90=%.3f max=%.3f m/s" % (
    float(np.mean(moving)) if moving.size else 0.0,
    float(np.percentile(moving, 90)) if moving.size else 0.0,
    float(np.max(moving)) if moving.size else 0.0,
))
PY

stop_pipeline
LP=
sleep 3

python3 /workspace/SimEnv/src/building_obstacles/scripts/evaluate_danger.py \
  --truth-file /workspace/SimEnv/results/danger_truth.json \
  --detected-file "$DETECTED" --output-file "$EVALUATION" >> "$RUN_LOG" 2>&1 || true

test -s "$RECORD" || { echo "ERROR: missing record" | tee -a "$RUN_LOG"; exit 4; }
test -s "$DETECTED" || { echo "ERROR: missing detection output" | tee -a "$RUN_LOG"; exit 5; }
test -s "$RESULTS/scanplanner_camera_fixed.mp4" || { echo "ERROR: missing camera video" | tee -a "$RUN_LOG"; exit 6; }
echo "runtime artifacts ready in $RESULTS" | tee -a "$RUN_LOG"
