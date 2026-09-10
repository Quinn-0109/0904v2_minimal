#!/usr/bin/env bash
# Native (non-Docker) entrypoint for the relocatable 0902overall bundle.
set -Eeuo pipefail
# minimal bundle: provide ROS env for fresh catkin rebuilds (genmsg etc.)
if [ -f /opt/ros/noetic/setup.bash ]; then
  set +u
  source /opt/ros/noetic/setup.bash
  set -u
fi

PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OVERALL_ROOT="$(cd "$PKG/.." && pwd)"
RUN_NAME="${1:-oblique_native_$(date +%Y%m%d_%H%M%S)}"
SEED_OFFSET="${2:-1}"

export SIMENV_ROOT="${SIMENV_ROOT:-$OVERALL_ROOT/SimEnv-master}"
export SIMENV_ASSET_ROOT="${SIMENV_ASSET_ROOT:-$OVERALL_ROOT/runtime_assets/SimEnv}"
export SCAN_WORKSPACE="${SCAN_WORKSPACE:-$PKG/work/mounts/SCAN-Planner}"
export RESULTS_ROOT="${RESULTS_ROOT:-$OVERALL_ROOT/results}"
export MISSION_CONFIG="${MISSION_CONFIG:-$PKG/work/mounts/SimEnv/src/simenv_bridge/config/three_floor_rl_mission.json}"
if [ -z "${SIMENV_TORCH_ROOT:-}" ] && [ -d "$OVERALL_ROOT/third_party/libtorch" ]; then
  export SIMENV_TORCH_ROOT="$OVERALL_ROOT/third_party/libtorch"
else
  export SIMENV_TORCH_ROOT="${SIMENV_TORCH_ROOT:-/opt/libtorch}"
fi
export SIMENV_VALIDATED_CONTROLLER="${SIMENV_VALIDATED_CONTROLLER:-$SIMENV_ASSET_ROOT/runtime_bin/junior_ctrl_validated}"
export SIMENV_NATIVE_BUILD_SPACE="${SIMENV_NATIVE_BUILD_SPACE:-$SIMENV_ROOT/.portable/build}"
export SIMENV_NATIVE_DEVEL_SPACE="${SIMENV_NATIVE_DEVEL_SPACE:-$SIMENV_ROOT/.portable/devel}"
export SCAN_BUILD_SPACE="${SCAN_BUILD_SPACE:-$OVERALL_ROOT/.portable_build/scanplanner_build_v2}"
export SCAN_DEVEL_SPACE="${SCAN_DEVEL_SPACE:-$OVERALL_ROOT/.portable_build/scanplanner_devel_v2}"
export THREE_FLOOR_SEED_OFFSET="$SEED_OFFSET"
export MAX_WALL_SEC="${MAX_WALL_SEC:-3000}"
# RGB-D camera topics remain enabled for online danger detection.  Only the
# optional MP4 encoder is disabled by default for the current acceptance run.
export RECORD_CAMERA_VIDEO="${RECORD_CAMERA_VIDEO:-false}"
export SIMENV_EXTRA_ROS_PACKAGE_PATH="$PKG/work/mounts/SimEnv/src:$SIMENV_ASSET_ROOT/src:$SIMENV_ROOT/src"
export ROS_PACKAGE_PATH="$SIMENV_EXTRA_ROS_PACKAGE_PATH${ROS_PACKAGE_PATH:+:$ROS_PACKAGE_PATH}"
export SIMENV_RUNTIME_ENV_FILE="$PKG/runtime_environment.sh"
export SIMENV_RUNTIME_CHECK_FILE="$PKG/check_runtime_dependencies.py"
source "$SIMENV_RUNTIME_ENV_FILE"

# The validated RL controller links the torch-wheel CUDA runtime by its short
# SONAME (libcublas.so.11, libcudart.so.11.0, libnvToolsExt.so.1).  The bundle
# originally shipped those as symlinks inside third_party/libtorch/lib; the
# image-provided /opt/libtorch carries the same hashed files but not the short
# names, and a silent fallback to the system CUDA toolkit build changes policy
# inference numerics enough to destabilise the gait.  Recreate the links from
# the hashed files and fail loudly when neither form is present.
ensure_torch_compat_link() {
  local dir="$1" short="$2" pattern="$3" candidate
  if [ -e "$dir/$short" ]; then
    return 0
  fi
  candidate="$(ls "$dir"/$pattern 2>/dev/null | head -n 1 || true)"
  if [ -n "$candidate" ]; then
    ln -s "${candidate##*/}" "$dir/$short"
    echo "[torch-compat] linked $short -> ${candidate##*/} in $dir"
    return 0
  fi
  echo "ERROR: $dir provides neither $short nor a matching $pattern." >&2
  echo "       The wheel-pinned CUDA runtime is required: a system toolkit" >&2
  echo "       substitute changes RL inference numerics and destabilises the gait." >&2
  exit 2
}
ensure_torch_compat_link "$SIMENV_TORCH_ROOT/lib" libcublas.so.11 'libcublas-*.so.11'
ensure_torch_compat_link "$SIMENV_TORCH_ROOT/lib" libcudart.so.11.0 'libcudart-*.so.11.0'
ensure_torch_compat_link "$SIMENV_TORCH_ROOT/lib" libnvToolsExt.so.1 'libnvToolsExt-*.so.1'

bundle_root="$(realpath -e "$OVERALL_ROOT")"
assert_bundle_path() {
  local label="$1"
  local configured="$2"
  local resolved
  resolved="$(realpath -m "$configured")"
  case "$resolved" in
    "$bundle_root"|"$bundle_root"/*) ;;
    *)
      echo "ERROR: $label must remain inside the unpacked 0902overall bundle: $resolved" >&2
      exit 2
      ;;
  esac
}
assert_bundle_path SIMENV_ROOT "$SIMENV_ROOT"
assert_bundle_path SIMENV_ASSET_ROOT "$SIMENV_ASSET_ROOT"
assert_bundle_path SCAN_WORKSPACE "$SCAN_WORKSPACE"
assert_bundle_path RESULTS_ROOT "$RESULTS_ROOT"
assert_bundle_path MISSION_CONFIG "$MISSION_CONFIG"
if [ "$SIMENV_TORCH_ROOT" != "/opt/libtorch" ]; then
  assert_bundle_path SIMENV_TORCH_ROOT "$SIMENV_TORCH_ROOT"
fi
assert_bundle_path SIMENV_VALIDATED_CONTROLLER "$SIMENV_VALIDATED_CONTROLLER"
assert_bundle_path SIMENV_NATIVE_BUILD_SPACE "$SIMENV_NATIVE_BUILD_SPACE"
assert_bundle_path SIMENV_NATIVE_DEVEL_SPACE "$SIMENV_NATIVE_DEVEL_SPACE"
assert_bundle_path SCAN_BUILD_SPACE "$SCAN_BUILD_SPACE"
assert_bundle_path SCAN_DEVEL_SPACE "$SCAN_DEVEL_SPACE"

mkdir -p "$RESULTS_ROOT" "$OVERALL_ROOT/.portable_build"

exec bash "$PKG/work/mounts/SimEnv/src/simenv_bridge/scripts/run_scanplanner_three_floor_rl.sh" "$RUN_NAME"
