#!/usr/bin/env bash
# Exploration-chain entrypoint for the 0904v2_minimal bundle.
# Usage: bash run_exploration.sh <RUN_NAME>     (single three-floor exploration run)
#
# This entry adds the FUEL-lite + FAST-LIO exploration chain on top of the
# bundle without touching the 0904 SCAN-Planner mission (run.sh stays the
# authority for that path).  Both entrypoints may be used in the same
# container, but never concurrently: each runner refuses to start while a
# live ROS/Gazebo session exists.
set -Eeuo pipefail
unset SIMENV_ROOT SIMENV_ASSET_ROOT SCAN_WORKSPACE RESULTS_ROOT MISSION_CONFIG \
      SIMENV_TORCH_ROOT SIMENV_VALIDATED_CONTROLLER \
      SIMENV_NATIVE_BUILD_SPACE SIMENV_NATIVE_DEVEL_SPACE \
      SCAN_BUILD_SPACE SCAN_DEVEL_SPACE ROS_PACKAGE_PATH THREE_FLOOR_SEED_OFFSET \
      GAZEBO_PLUGIN_PATH GAZEBO_MODEL_PATH
export MAX_WALL_SEC="${MAX_WALL_SEC:-3600}"
if [ -f /opt/ros/noetic/setup.bash ]; then
  set +u
  source /opt/ros/noetic/setup.bash
  set -u
fi
BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$BUNDLE_ROOT/mission/work/mounts/ExplorationWS/scripts/run_exploration_three_floor.sh" "$@"
