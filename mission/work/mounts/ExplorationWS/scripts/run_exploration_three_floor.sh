#!/usr/bin/env bash
# Three-floor exploration chain (FUEL-lite + FAST-LIO + stair transitions)
# migrated into the 0904v2_minimal bundle.  Follows the bundle's runner
# conventions: environment isolation, task-owned Xvfb, own build/devel spaces,
# never mutates SimEnv-master / runtime_assets / the 0904 mission mounts.
#
# Mission (same chain as the standalone simenv_exploration delivery):
#   F1 explore -> F1->F2 climb -> F2 explore -> F2->F3 climb -> F3 explore
#   -> descent return F3->F2->F1
# Success = logs/stair_descent.json phase=FIRST_FLOOR_RETURNED.
#
# Usage: run_exploration_three_floor.sh <RUN_NAME>   (results under <bundle>/results/<RUN_NAME>)
set -Eeuo pipefail

PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"            # ExplorationWS mount
WORKSPACE="$PKG"                                                   # launch tree root (src/ + generated_building/)
BUNDLE_ROOT="$(cd "$PKG/../../../.." && pwd)"                      # 0904v2_minimal
SIMENV_ASSET_ROOT="${SIMENV_ASSET_ROOT:-$BUNDLE_ROOT/runtime_assets/SimEnv}"
SIMENV_MASTER_ROOT="${SIMENV_MASTER_ROOT:-$BUNDLE_ROOT/SimEnv-master}"
SIMENV_TORCH_ROOT="${SIMENV_TORCH_ROOT:-/opt/libtorch}"
RESULTS_ROOT="${RESULTS_ROOT:-$BUNDLE_ROOT/results}"
RUN_NAME="${1:-exploration_$(date +%Y%m%d_%H%M%S)}"
MAX_WALL_SEC="${MAX_WALL_SEC:-3600}"
EXPL_BUILD_SPACE="${EXPL_BUILD_SPACE:-$BUNDLE_ROOT/.portable_build/exploration_build}"
EXPL_DEVEL_SPACE="${EXPL_DEVEL_SPACE:-$BUNDLE_ROOT/.portable_build/exploration_devel}"
XVFB_DISPLAY="${EXPLORATION_XVFB_DISPLAY:-:99}"

export SIMENV_EXPLORATION_WORKSPACE="$WORKSPACE"
# Validated controller + gazebo plugins live in the bundle's prebuilt devel.
export SIMENV_EXPL_DEVEL_SPACE="$SIMENV_MASTER_ROOT/.portable/devel"

case "$RUN_NAME" in
  .|..) echo "ERROR: unsafe run name: $RUN_NAME" >&2; exit 2 ;;
esac
if ! [[ "$RUN_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "ERROR: run name must match [A-Za-z0-9][A-Za-z0-9._-]*" >&2; exit 2
fi

for required in \
  "$SIMENV_ASSET_ROOT/src/unitree_guide/unitree_guide/unitree_guide/launch/multi_floor_gazeboSim.launch" \
  "$SIMENV_ASSET_ROOT/src/unitree_guide/logs/policy_act_inference_plane.pt" \
  "$SIMENV_ASSET_ROOT/src/unitree_guide/logs/policy_act_inference_stair.pt" \
  "$SIMENV_EXPL_DEVEL_SPACE/lib/unitree_guide/junior_ctrl" \
  "$SIMENV_EXPL_DEVEL_SPACE/lib/liblivox_laser_simulation.so" \
  "$SIMENV_EXPL_DEVEL_SPACE/lib/libunitree_legged_control.so" \
  "$WORKSPACE/generated_building/elevator_three_floor_debug/competition_scene_with_dangers.world" \
  "$WORKSPACE/src/simenv_exploration/launch/fuel_semantic_fastlio_exploration.launch"; do
  [ -e "$required" ] || { echo "ERROR: bundle asset missing: $required" >&2; exit 2; }
done

# Match process names/paths only: a bare substring match on full args also
# hits innocent processes whose command line merely mentions "roslaunch"
# (e.g. `tail -F .../step_roslaunch.log`).
if ps -eo stat=,comm=,args= | awk \
    '$1 !~ /^Z/ && ($2 ~ /^(rosmaster|roscore|gzserver|gzclient)$/ ||
                    $0 ~ /(^|\/)roslaunch /) { found=1 }
     END { exit !found }'; then
  echo "ERROR: live ROS/Gazebo session already exists; refusing to share it" >&2
  echo "       (protects the 0904 mission and this run from cross-cleanup)" >&2
  exit 2
fi

# --- ROS environment -------------------------------------------------------
set +u
source /opt/ros/noetic/setup.bash
set -u

# The validated RL controller links the torch-wheel CUDA runtime by short
# SONAME (same requirement as mission/run_native.sh); recreate the links from
# the hashed files and fail loudly when neither form is present.
ensure_torch_compat_link() {
  local dir="$1" short="$2" pattern="$3" candidate
  if [ -e "$dir/$short" ]; then return 0; fi
  candidate="$(ls "$dir"/$pattern 2>/dev/null | head -n 1 || true)"
  if [ -n "$candidate" ]; then
    ln -s "${candidate##*/}" "$dir/$short"
    echo "[exploration] linked $short -> ${candidate##*/} in $dir"
    return 0
  fi
  echo "ERROR: $dir provides neither $short nor a matching $pattern." >&2
  exit 2
}
ensure_torch_compat_link "$SIMENV_TORCH_ROOT/lib" libcublas.so.11 'libcublas-*.so.11'
ensure_torch_compat_link "$SIMENV_TORCH_ROOT/lib" libcudart.so.11.0 'libcudart-*.so.11.0'
ensure_torch_compat_link "$SIMENV_TORCH_ROOT/lib" libnvToolsExt.so.1 'libnvToolsExt-*.so.1'

# --- build the exploration mount (both packages; no other tree is touched) --
mkdir -p "$EXPL_BUILD_SPACE" "$EXPL_DEVEL_SPACE"
catkin_make -C "$WORKSPACE" --build "$EXPL_BUILD_SPACE" \
  -j2 -DCMAKE_BUILD_TYPE=Release \
  -DPYTHON_EXECUTABLE=/usr/bin/python3 \
  -DCATKIN_DEVEL_PREFIX="$EXPL_DEVEL_SPACE" \
  >"$EXPL_BUILD_SPACE/build.log" 2>&1 || {
    echo "ERROR: exploration workspace build failed" >&2
    tail -n 120 "$EXPL_BUILD_SPACE/build.log" >&2
    exit 2
  }

set +u
source "$SIMENV_MASTER_ROOT/.portable/devel/setup.bash" --extend   # prebuilt controller/door/gazebo-plugin nodes
source "$EXPL_DEVEL_SPACE/setup.bash" --extend                     # simenv_exploration + fastlio artifacts
set -u

# Package resolution: exploration mount first, then the bundle's support trees
# (read-only reuse — exactly the order the 0904 mission uses for unitree_guide).
export ROS_PACKAGE_PATH="$WORKSPACE/src:$SIMENV_ASSET_ROOT/src:$SIMENV_MASTER_ROOT/src${ROS_PACKAGE_PATH:+:$ROS_PACKAGE_PATH}"
export GAZEBO_PLUGIN_PATH="$SIMENV_MASTER_ROOT/.portable/devel/lib${GAZEBO_PLUGIN_PATH:+:$GAZEBO_PLUGIN_PATH}"
export GAZEBO_MODEL_PATH="${GAZEBO_MODEL_PATH:-}:$WORKSPACE/generated_building:$SIMENV_ASSET_ROOT/src/unitree_guide/unitree_ros/unitree_gazebo/models"
export LD_LIBRARY_PATH="$SIMENV_TORCH_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

rospack profile >/dev/null
for p in simenv_exploration fast_lio unitree_guide a1_description unitree_controller; do
  rospack find "$p" >/dev/null || { echo "ERROR: ROS package not resolvable: $p" >&2; exit 2; }
done
python3 -c 'import roslib.packages; assert roslib.packages.find_node("simenv_exploration", "classic_trotting_controller.py"), "controller wrapper not discoverable"' \
  || exit 2
python3 -c 'import roslib.packages; assert roslib.packages.find_node("simenv_exploration", "fuel_lite_planner"), "fuel_lite_planner not built"' \
  || exit 2
python3 - <<'PY' || exit 2
import roslaunch, roslaunch.rlutil, roslib.packages
path = roslaunch.rlutil.resolve_launch_arguments(
    ["simenv_exploration", "fuel_semantic_fastlio_exploration.launch"])[0]
config = roslaunch.config.load_config_default([path], None)
missing = [n.package + "/" + n.type for n in config.nodes
           if not roslib.packages.find_node(n.package, n.type)]
assert not missing, "missing ROS node executables: " + ", ".join(missing)
print("[exploration] launch graph ok: {} nodes".format(len(config.nodes)))
PY

# Preflight mode: stop after build + launch-graph validation (0904 convention).
if [ "${EXPLORATION_PREFLIGHT_ONLY:-0}" = "1" ]; then
  echo "[exploration] PREFLIGHT PASS: build, package resolution, node executables,"
  echo "[exploration]               and launch graph validated; no simulation started"
  exit 0
fi

# --- results directory ------------------------------------------------------
RESULTS_DIR="$RESULTS_ROOT/$RUN_NAME"
if [ -d "$RESULTS_DIR" ] && find "$RESULTS_DIR" -mindepth 1 -print -quit | grep -q .; then
  echo "ERROR: result directory is not empty: $RESULTS_DIR" >&2; exit 2
fi
mkdir -p "$RESULTS_DIR/logs"
: > "$RESULTS_DIR/summary.txt"
RUN_LOG="$RESULTS_DIR/step_roslaunch.log"

# --- task-owned Xvfb (Gazebo camera sensors need an X context) ---------------
if DISPLAY="$XVFB_DISPLAY" xdpyinfo >/dev/null 2>&1; then
  echo "ERROR: X display $XVFB_DISPLAY is already active; refusing to share it" >&2
  exit 2
fi
setsid Xvfb "$XVFB_DISPLAY" -screen 0 1280x720x24 -nolisten tcp \
  >"$RESULTS_DIR/xvfb.log" 2>&1 &
XVFB_PID=$!
export DISPLAY="$XVFB_DISPLAY"
for _ in $(seq 1 50); do
  DISPLAY="$XVFB_DISPLAY" xdpyinfo >/dev/null 2>&1 && break
  kill -0 "$XVFB_PID" 2>/dev/null || break
  sleep 0.1
done
if ! kill -0 "$XVFB_PID" 2>/dev/null || \
   ! DISPLAY="$XVFB_DISPLAY" xdpyinfo >/dev/null 2>&1; then
  echo "ERROR: task-owned Xvfb failed to become ready on $XVFB_DISPLAY" >&2
  exit 2
fi
echo "[exploration] task-owned Xvfb ready on $XVFB_DISPLAY"

# --- run the mission ---------------------------------------------------------
LAUNCH_PID=""
stop_launch() {
  local pid="${LAUNCH_PID:-}"
  [ -n "$pid" ] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    kill -INT -- "-$pid" 2>/dev/null || true
    for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  fi
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM -- "-$pid" 2>/dev/null || true
    for _ in $(seq 1 5); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  fi
  wait "$pid" 2>/dev/null || true
  LAUNCH_PID=""
}
stop_xvfb() {
  local pid="${XVFB_PID:-}"
  [ -n "$pid" ] || return 0
  kill -TERM -- "-$pid" 2>/dev/null || true
  for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done
  wait "$pid" 2>/dev/null || true
  XVFB_PID=""
}
stop_runtime() { stop_launch; stop_xvfb; }
trap stop_runtime EXIT INT TERM

echo "[exploration] results: $RESULTS_DIR"
setsid roslaunch simenv_exploration fuel_semantic_fastlio_exploration.launch \
  output_dir:="$RESULTS_DIR" \
  enable_third_floor_descent:=true \
  descent_speed_mps:=0.80 \
  stair_ascent_speed_third_mps:=0.70 \
  pre_ascent_stand_seconds:=0.0 \
  landing_heading_tolerance_rad:=0.05 \
  truth_flight_b_max_yaw_rate_rps:=0.18 \
  truth_flight_b_recovery_hold_seconds:=6.0 \
  >"$RUN_LOG" 2>&1 &
LAUNCH_PID=$!
echo "[exploration] roslaunch pid=$LAUNCH_PID"

read_phase() {
  python3 -c "
import json
try:
    d=json.load(open('$1'))
    print(d.get('phase','?'))
except Exception:
    print('PARTIAL')" 2>/dev/null
}

wait_phase() {  # $1=json $2=budget_sec -> echoes phase, returns 1 on NO_TRACE
  local waited=0 phase="WAIT"
  while [ "$waited" -lt "$2" ]; do
    if [ -f "$1" ]; then
      phase="$(read_phase "$1")"
      if [ "$phase" != "?" ] && [ "$phase" != "PARTIAL" ]; then
        echo "$phase"; return 0
      fi
    fi
    if ! kill -0 "$LAUNCH_PID" 2>/dev/null; then break; fi
    sleep 5
    waited=$((waited + 5))
    if [ $((waited % 120)) -eq 0 ]; then
      echo "[exploration] waiting for $1 (${waited}s)" >&2
    fi
  done
  if [ -f "$1" ]; then echo "$(read_phase "$1")"; else echo "NO_TRACE"; fi
}

T12_FAIL_PHASES="STAIR_ASCENT_TIMEOUT STAIR_PRE_ASCENT_ALIGN_TIMEOUT STAIR_FLIGHT_A_FALL_DETECTED STAIR_FLIGHT_A_ALIGNMENT_LOST STAIR_FLIGHT_B_FALL_DETECTED STAIR_LANDING_FALL_DETECTED STAIR_LANDING_TIMEOUT STAIR_POLICY_TIMEOUT STAIR_ENTRY_NOT_REACHED STAIR_HANDOFF_NOT_REACHED STAIR_SCENE_NO_TWO_FLIGHT_GEOMETRY SECOND_FLOOR_HANDOFF_TIMEOUT"
F3_OK_TERMS="STAIR_CORRIDOR_EXIT_HANDOFF STAIR_CORRIDOR_EXIT_HANDOFF_TRUTH_GATE STAIR_LOBBY_HANDOFF_LOCALIZATION_FALLBACK STAIR_WAIT_ZONE_REACHED"
STARTED_AT="$(date +%s)"
MISSION_OK=0

PH12="$(wait_phase "$RESULTS_DIR/logs/stair_transition.json" 1500)"
echo "[exploration] F1->F2 phase: $PH12"

PH23="-"; F3TERM=""; DESC_PH="-"
if ! [[ " $T12_FAIL_PHASES " == *" $PH12 "* ]]; then
  PH23="$(wait_phase "$RESULTS_DIR/logs/second_to_third_floor_stair_transition.json" 1500)"
  echo "[exploration] F2->F3 phase: $PH23"
  if [ "$PH23" = "THIRD_FLOOR_HANDOFF_COMPLETE" ]; then
    # F3 terminal state: third_floor_handoff.json is authoritative; fall back
    # to the periodic baseline_summary.json snapshot.
    f3_waited=0
    while [ "$f3_waited" -lt 500 ]; do
      F3TERM="$(python3 -c "
import json
for p in ('$RESULTS_DIR/third_floor_handoff.json', '$RESULTS_DIR/third_floor/baseline_summary.json'):
    try:
        d=json.load(open(p))
        t=d.get('termination_reason') or ''
        st=d.get('state') or ''
        if 'EXPLORATION' in st and 'START' in st:
            continue
        print(t); break
    except Exception:
        pass
else:
    print('')" 2>/dev/null)"
      [ -n "$F3TERM" ] && [ "$F3TERM" != "running" ] && break
      kill -0 "$LAUNCH_PID" 2>/dev/null || break
      sleep 5
      f3_waited=$((f3_waited + 5))
    done
    echo "[exploration] third_floor termination: ${F3TERM:-ABSENT}"
    if [[ " $F3_OK_TERMS " == *" $F3TERM "* ]]; then
      DESC_PH="$(wait_phase "$RESULTS_DIR/logs/stair_descent.json" 900)"
      echo "[exploration] descent phase: $DESC_PH"
    else
      DESC_PH="F3_EXPLORATION_FAILED"
    fi
  fi
else
  echo "[exploration] F1->F2 failed ($PH12); skipping later stages"
fi

if [ "$PH23" = "THIRD_FLOOR_HANDOFF_COMPLETE" ] && [ "$DESC_PH" = "FIRST_FLOOR_RETURNED" ]; then
  MISSION_OK=1
fi

# Overall wall cap (covers fastlio-lockup / manager-deadlock tails).
NOW="$(date +%s)"
if [ $((NOW - STARTED_AT)) -ge "$MAX_WALL_SEC" ]; then
  echo "[exploration] ERROR: wall timeout after ${MAX_WALL_SEC}s" >&2
fi

stop_runtime
trap - EXIT INT TERM
sleep 3

echo "run=$RUN_NAME F1F2=$PH12 F2F3=$PH23 F3=${F3TERM:--} descent=$DESC_PH mission=$MISSION_OK" \
  >> "$RESULTS_DIR/summary.txt"
grep -E "STPUB|DSTPUB|Flight-B stalled|Descent stall|signal_shutdown|MISSION|handoff" \
  "$RUN_LOG" | tail -25 >> "$RESULTS_DIR/summary.txt" || true

echo "[exploration] mission=$MISSION_OK (F1->F2=$PH12 F2->F3=$PH23 descent=$DESC_PH)"
echo "[exploration] log: $RUN_LOG"
[ "$MISSION_OK" = "1" ] || exit 1
