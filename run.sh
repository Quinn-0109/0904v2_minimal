#!/usr/bin/env bash
# Single entrypoint. Usage: bash run.sh <RUN_NAME> [SEED_OFFSET]   (seed 20 = validated fixed layout)
set -Eeuo pipefail
unset SIMENV_ROOT SIMENV_ASSET_ROOT SCAN_WORKSPACE RESULTS_ROOT MISSION_CONFIG \
      SIMENV_TORCH_ROOT SIMENV_VALIDATED_CONTROLLER \
      SIMENV_NATIVE_BUILD_SPACE SIMENV_NATIVE_DEVEL_SPACE \
      SCAN_BUILD_SPACE SCAN_DEVEL_SPACE ROS_PACKAGE_PATH THREE_FLOOR_SEED_OFFSET
export MAX_WALL_SEC="${MAX_WALL_SEC:-3000}"
export RECORD_CAMERA_VIDEO="${RECORD_CAMERA_VIDEO:-false}"
if [ -f /opt/ros/noetic/setup.bash ]; then
  set +u
  source /opt/ros/noetic/setup.bash
  set -u
fi
BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$BUNDLE_ROOT/mission/run_native.sh" "$@"
