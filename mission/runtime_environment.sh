#!/usr/bin/env bash
# Source after catkin setup files: register relocated libraries without executing
# archived catkin Python relays that still contain /workspace/0904_v2 paths.
# gazebo_ros/gzserver sources this setup before starting the real server. Use
# the same library paths for the earlier dependency check (camera base plugins
# are not necessarily in /opt/ros/noetic/lib or the system ldconfig cache).
simenv_gazebo_environment() {
  local prefix setup_path original_master original_database restore_nounset=0
  command -v pkg-config >/dev/null 2>&1 || return 0
  prefix="$(pkg-config --variable=prefix gazebo 2>/dev/null)" || return 0
  setup_path="$prefix/share/gazebo/setup.sh"
  [ -f "$setup_path" ] || return 0
  original_master="${GAZEBO_MASTER_URI:-}"
  original_database="${GAZEBO_MODEL_DATABASE_URI:-}"
  case "$-" in *u*) restore_nounset=1 ;; esac
  set +u
  source "$setup_path"
  if [ "$restore_nounset" = 1 ]; then set -u; fi
  if [ -n "$original_master" ]; then export GAZEBO_MASTER_URI="$original_master"; fi
  if [ -n "$original_database" ]; then export GAZEBO_MODEL_DATABASE_URI="$original_database"; fi
  return 0
}
simenv_gazebo_environment
unset -f simenv_gazebo_environment
SIMENV_RUNTIME_PREFIX="${SIMENV_NATIVE_DEVEL_SPACE:-$SIMENV_ROOT/.portable/devel}"
export CMAKE_PREFIX_PATH="$SIMENV_RUNTIME_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export LD_LIBRARY_PATH="$SIMENV_TORCH_ROOT/lib:$SIMENV_RUNTIME_PREFIX/lib:/opt/ros/noetic/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBTORCH_LIBRARY_PATH="$SIMENV_TORCH_ROOT/lib"
export GAZEBO_PLUGIN_PATH="$SIMENV_RUNTIME_PREFIX/lib${GAZEBO_PLUGIN_PATH:+:$GAZEBO_PLUGIN_PATH}"
# The inner source packages must precede archived generated relay packages.
# Generated messages/services and ROS's own Python packages remain available.
export PYTHONPATH="$SIMENV_ASSET_ROOT/src/building_generator_classic:$SIMENV_ASSET_ROOT/src/building_generator_core:$SIMENV_ROOT/src/building_generator_classic:$SIMENV_ROOT/src/building_generator_core:$SIMENV_RUNTIME_PREFIX/lib/python3/dist-packages:/opt/ros/noetic/lib/python3/dist-packages${PYTHONPATH:+:$PYTHONPATH}"
