#!/usr/bin/env bash
set -Eeuo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
failures=0

ok() { printf '[OK] %s\n' "$1"; }
bad() { printf '[MISSING/INCOMPATIBLE] %s\n' "$1" >&2; failures=$((failures + 1)); }

if [ -f /opt/ros/noetic/setup.bash ]; then
  set +u
  source /opt/ros/noetic/setup.bash
  set -u
fi

if [ "$(uname -m)" = x86_64 ]; then ok 'x86_64 architecture'; else bad "x86_64 required (found $(uname -m))"; fi
if [ -r /etc/os-release ] && grep -q '^VERSION_ID="20.04"' /etc/os-release; then
  ok 'Ubuntu 20.04'
else
  bad 'Ubuntu 20.04 is the validated operating system'
fi

for command_name in bash python3 gcc g++ cmake make catkin_make roslaunch rospack gzserver nvidia-smi; do
  if command -v "$command_name" >/dev/null 2>&1; then ok "command: $command_name"; else bad "command: $command_name"; fi
done

if [ -f /opt/ros/noetic/setup.bash ]; then ok 'ROS Noetic'; else bad '/opt/ros/noetic/setup.bash'; fi
torch_runtime_ok=0
torch_runtime_root=""
for candidate in "${SIMENV_TORCH_ROOT:-}" "$BUNDLE_ROOT/third_party/libtorch" "/opt/libtorch"; do
  [ -n "$candidate" ] || continue
  if [ -f "$candidate/lib/libcublas.so.11" ] && \
     [ -f "$candidate/lib/libcudart.so.11.0" ] && \
     [ -f "$candidate/lib/libnvToolsExt.so.1" ]; then
    torch_runtime_ok=1
    torch_runtime_root="$candidate"
    break
  fi
done
if [ -d /usr/local/cuda/lib64 ]; then
  ok 'CUDA runtime directory'
elif [ "$torch_runtime_ok" -eq 1 ]; then
  ok "CUDA runtime supplied by LibTorch: $torch_runtime_root"
else
  bad 'CUDA runtime: neither /usr/local/cuda/lib64 nor complete LibTorch runtime'
fi

python3 - <<'PY' >/dev/null 2>&1 && ok 'required Python modules' || bad 'Python modules: numpy, cv2, yaml, matplotlib'
import cv2, matplotlib, numpy, yaml
PY

required_files=(
  "$BUNDLE_ROOT/mission/run_native.sh"
  "$BUNDLE_ROOT/SimEnv-master/src/unitree_guide/unitree_guide/unitree_guide/CMakeLists.txt"
  "$BUNDLE_ROOT/mission/work/mounts/SCAN-Planner/src/planner/plan_manage/package.xml"
  "$BUNDLE_ROOT/runtime_assets/SimEnv/generated_building/elevator_three_floor_debug/competition_scene_with_dangers.world"
  "$BUNDLE_ROOT/runtime_assets/SimEnv/generated_building/elevator_three_floor_debug/layout_metadata.json"
  "$BUNDLE_ROOT/runtime_assets/SimEnv/generated_building/elevator_three_floor_debug/model.sdf"
  "$BUNDLE_ROOT/runtime_assets/SimEnv/src/simenv_competitor/scripts/controller_mode_bootstrap.py"
  "$BUNDLE_ROOT/runtime_assets/SimEnv/runtime_bin/junior_ctrl_validated"
  "$BUNDLE_ROOT/runtime_assets/SimEnv/src/unitree_guide/logs/policy_act_inference_plane.pt"
  "$BUNDLE_ROOT/runtime_assets/SimEnv/src/unitree_guide/logs/policy_act_inference_stair.pt"
  "$BUNDLE_ROOT/third_party/libtorch/share/cmake/Torch/TorchConfig.cmake"
  "$BUNDLE_ROOT/third_party/libtorch/lib/libtorch.so"
  "$BUNDLE_ROOT/third_party/libtorch/lib/libtorch_cpu.so"
  "$BUNDLE_ROOT/third_party/libtorch/lib/libtorch_cuda.so"
)
for required in "${required_files[@]}"; do
  rel=${required#"$BUNDLE_ROOT"/}
  stripped=${rel#third_party/libtorch/}
  if [ -f "$required" ]; then
    ok "bundle file: $rel"
  elif [ "$stripped" != "$rel" ] && [ -f "/opt/libtorch/$stripped" ]; then
    ok "libtorch (image): /opt/libtorch/$stripped"
  else
    bad "bundle file: $rel"
  fi
done

available_kb="$(df -Pk "$BUNDLE_ROOT" | awk 'NR==2 {print $4}')"
if [ "${available_kb:-0}" -ge 1048576 ]; then
  ok 'at least 1 GiB free for builds and results'
else
  bad 'at least 1 GiB free for builds and results'
fi

if [ "$failures" -ne 0 ]; then
  printf 'Preflight failed with %d problem(s).\n' "$failures" >&2
  exit 2
fi
printf 'Preflight passed. Bundle root: %s\n' "$BUNDLE_ROOT"
