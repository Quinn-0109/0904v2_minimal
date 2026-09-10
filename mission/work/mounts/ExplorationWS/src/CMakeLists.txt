# Windows archives do not preserve the usual catkin CMakeLists.txt symlink.
# Include the canonical Noetic workspace entry point from a regular file so
# this source tree can also be mounted read-only in the mission container.
cmake_minimum_required(VERSION 3.0.2)
project(SCANPlannerWorkspace)
include("/opt/ros/noetic/share/catkin/cmake/toplevel.cmake")
