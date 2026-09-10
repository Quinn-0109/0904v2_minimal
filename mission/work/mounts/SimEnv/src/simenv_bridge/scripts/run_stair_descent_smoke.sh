#!/usr/bin/env bash
# Run only the real A1 stair-RL F3 -> F2 -> F1 return leg.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SIMENV_ROOT="${SIMENV_ROOT:-$(cd "$PACKAGE_DIR/../.." && pwd)}"
MISSION_CONFIG="${MISSION_CONFIG:-$PACKAGE_DIR/config/three_floor_rl_mission.json}"
MISSION_CHECKER="$SCRIPT_DIR/check_three_floor_rl_mission.py"
SMOKE_CHECKER="$SCRIPT_DIR/check_stair_descent_smoke.py"
RUN_NAME="${1:-stair_descent_smoke_$(date +%Y%m%d_%H%M%S)}"
MAX_WALL_SEC="${MAX_WALL_SEC:-1200}"
DESCENT_SPEED_MPS="${DESCENT_SPEED_MPS:-0.80}"

case "$RUN_NAME" in
  .|..) echo "ERROR: unsafe run name: $RUN_NAME" >&2; exit 2 ;;
esac
if [[ ! "$RUN_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "ERROR: run name must match [A-Za-z0-9][A-Za-z0-9._-]*" >&2
  exit 2
fi
if ! [[ "$MAX_WALL_SEC" =~ ^[0-9]+([.][0-9]+)?$ ]] || \
   ! awk -v value="$MAX_WALL_SEC" 'BEGIN { exit !(value > 0) }'; then
  echo "ERROR: MAX_WALL_SEC must be positive" >&2
  exit 2
fi
if ! [[ "$DESCENT_SPEED_MPS" =~ ^[0-9]+([.][0-9]+)?$ ]] || \
   ! awk -v value="$DESCENT_SPEED_MPS" \
      'BEGIN { exit !(value >= 0.30 && value <= 0.80) }'; then
  echo "ERROR: DESCENT_SPEED_MPS must be in [0.30, 0.80]" >&2
  exit 2
fi

if [ -z "${ROS_DISTRO:-}" ]; then
  set +u
  source /opt/ros/noetic/setup.bash
  if [ -f "$SIMENV_ROOT/devel/setup.bash" ]; then
    source "$SIMENV_ROOT/devel/setup.bash"
  fi
  if [ -f "$SIMENV_ROOT/devel_0803b/setup.bash" ]; then
    source "$SIMENV_ROOT/devel_0803b/setup.bash" --extend
  fi
  set -u
fi
export ROS_PACKAGE_PATH="$PACKAGE_DIR${ROS_PACKAGE_PATH:+:$ROS_PACKAGE_PATH}"

test -x "$MISSION_CHECKER" || {
  echo "ERROR: mission checker is not executable: $MISSION_CHECKER" >&2
  exit 2
}
test -x "$SMOKE_CHECKER" || {
  echo "ERROR: smoke checker is not executable: $SMOKE_CHECKER" >&2
  exit 2
}
python3 "$MISSION_CHECKER" --config "$MISSION_CONFIG" --validate-config \
  --require-runtime-files >/dev/null
rospack find simenv_competitor >/dev/null
rospack find unitree_guide >/dev/null
rospack find simenv_bridge >/dev/null
python3 -c 'import roslaunch,roslaunch.rlutil,roslib.packages; path=roslaunch.rlutil.resolve_launch_arguments(["simenv_bridge","stair_descent_physical_smoke.launch"])[0]; config=roslaunch.config.load_config_default([path],None); missing=[node.package+"/"+node.type for node in config.nodes if not roslib.packages.find_node(node.package,node.type)]; assert not missing, "missing ROS node executables: "+", ".join(missing); print("validated {} descent-smoke node executables".format(len(config.nodes)))'

# The enclosing Docker runner creates a clean, owned container.  Never clean
# unknown processes if that invariant is violated.
if ps -eo stat=,args= | awk \
    '$1 !~ /^Z/ && $0 ~ /[r]osmaster|[g]zserver|[r]oslaunch/ { found=1 }
     END { exit !found }'; then
  echo "ERROR: live ROS/Gazebo session already exists; refusing broad cleanup" >&2
  exit 2
fi

RESULTS_DIR="$SIMENV_ROOT/results/$RUN_NAME"
if [ -d "$RESULTS_DIR" ] && find "$RESULTS_DIR" -mindepth 1 -print -quit | grep -q .; then
  echo "ERROR: result directory is not empty: $RESULTS_DIR" >&2
  exit 2
fi
mkdir -p "$RESULTS_DIR"
RUN_LOG="$RESULTS_DIR/roslaunch.log"

WORLD_FILE="${THREE_FLOOR_WORLD_FILE:-$(python3 "$MISSION_CHECKER" --config "$MISSION_CONFIG" --emit-runtime world_file)}"
LAYOUT_METADATA="${THREE_FLOOR_LAYOUT_METADATA:-$(python3 "$MISSION_CHECKER" --config "$MISSION_CONFIG" --emit-runtime layout_metadata)}"
STAIR_MODEL_SDF="${THREE_FLOOR_STAIR_MODEL_SDF:-$(python3 "$MISSION_CHECKER" --config "$MISSION_CONFIG" --emit-runtime stair_model_sdf)}"
PLANE_POLICY="$(python3 "$MISSION_CHECKER" --config "$MISSION_CONFIG" --emit-runtime plane_policy)"
STAIR_POLICY="$(python3 "$MISSION_CHECKER" --config "$MISSION_CONFIG" --emit-runtime stair_policy)"

LAUNCH_PID=""
stop_launch() {
  local pid="${LAUNCH_PID:-}"
  [ -n "$pid" ] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    kill -INT -- "-$pid" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
  fi
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM -- "-$pid" 2>/dev/null || true
    for _ in $(seq 1 5); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
  fi
  wait "$pid" 2>/dev/null || true
  LAUNCH_PID=""
}
trap stop_launch EXIT INT TERM

echo "[descent-smoke] results: $RESULTS_DIR"
echo "[descent-smoke] physical gait: $STAIR_POLICY"
echo "[descent-smoke] commanded stair speed: ${DESCENT_SPEED_MPS} m/s"
echo "[descent-smoke] spawn: production F3 handoff (-2.665, 1.625, 5.71)"
if [ -n "${THREE_FLOOR_SCENE_MANIFEST:-}" ]; then
  cp "$THREE_FLOOR_SCENE_MANIFEST" "$RESULTS_DIR/scene_preparation.json"
fi

setsid roslaunch simenv_bridge stair_descent_physical_smoke.launch \
  output_dir:="$RESULTS_DIR" \
  world_file:="$WORLD_FILE" \
  layout_metadata:="$LAYOUT_METADATA" \
  stair_model_sdf:="$STAIR_MODEL_SDF" \
  plane_policy:="$PLANE_POLICY" \
  stair_policy:="$STAIR_POLICY" \
  descent_speed_mps:="$DESCENT_SPEED_MPS" \
  >"$RUN_LOG" 2>&1 &
LAUNCH_PID=$!

TERMINAL="$RESULTS_DIR/logs/stair_descent.json"
STARTED_AT="$(date +%s)"
STATUS="running"
while kill -0 "$LAUNCH_PID" 2>/dev/null; do
  if [ -s "$TERMINAL" ]; then
    PHASE="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("phase","unknown"))' "$TERMINAL" 2>/dev/null || echo partial)"
    if [ "$PHASE" != "WAIT_F3" ] && [ "$PHASE" != "partial" ]; then
      STATUS="$PHASE"
      break
    fi
  fi
  NOW="$(date +%s)"
  if awk -v now="$NOW" -v start="$STARTED_AT" -v limit="$MAX_WALL_SEC" \
      'BEGIN { exit !((now-start) >= limit) }'; then
    STATUS="runner_timeout"
    echo "[descent-smoke] ERROR: wall timeout after ${MAX_WALL_SEC}s" >&2
    break
  fi
  sleep 2
done

stop_launch
trap - EXIT INT TERM
sleep 3

set +e
python3 "$SMOKE_CHECKER" \
  --results "$RESULTS_DIR" \
  --expected-policy-basename "$(basename "$STAIR_POLICY")" \
  --output "$RESULTS_DIR/stair_descent_physical_smoke_acceptance.json"
CHECK_RC=$?
set -e
if [ "$CHECK_RC" -ne 0 ]; then
  echo "[descent-smoke] terminal: $STATUS" >&2
  echo "[descent-smoke] acceptance failed; inspect $RUN_LOG" >&2
  exit "$CHECK_RC"
fi
echo "[descent-smoke] PASS: physical stair RL returned from F3 to F1"
