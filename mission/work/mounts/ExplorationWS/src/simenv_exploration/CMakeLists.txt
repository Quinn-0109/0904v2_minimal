cmake_minimum_required(VERSION 3.0.2)
project(simenv_exploration)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  nav_msgs
  roscpp
  sensor_msgs
  std_msgs
)
find_package(octomap REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)
set(octomap_INCLUDE_DIRS ${OCTOMAP_INCLUDE_DIRS})
set(octomap_LIBRARIES ${OCTOMAP_LIBRARIES})

add_service_files(FILES CheckTwinCylinder.srv)
generate_messages(DEPENDENCIES std_msgs)

catkin_package(
  CATKIN_DEPENDS message_runtime nav_msgs roscpp sensor_msgs std_msgs
  DEPENDS octomap
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OCTOMAP_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

# Exploration-chain C++ nodes only (the octomap_* offline inspectors were not
# part of the online chain and were dropped from this package).
add_executable(fastlio_voxel_mapper src/fastlio_voxel_mapper.cpp)
add_dependencies(fastlio_voxel_mapper ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_compile_features(fastlio_voxel_mapper PRIVATE cxx_std_14)
target_link_libraries(fastlio_voxel_mapper ${catkin_LIBRARIES} ${OCTOMAP_LIBRARIES})
add_executable(fuel_lite_planner src/fuel_lite_planner.cpp)
add_dependencies(fuel_lite_planner ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_compile_features(fuel_lite_planner PRIVATE cxx_std_14)
target_link_libraries(fuel_lite_planner
  ${catkin_LIBRARIES} ${OCTOMAP_LIBRARIES} ${OpenCV_LIBRARIES})

install(TARGETS fastlio_voxel_mapper fuel_lite_planner
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

catkin_install_python(PROGRAMS
  scripts/xvfb_display_guard.py
  scripts/controller_mode_bootstrap.py
  scripts/fastlio_pointcloud_adapter.py
  scripts/manual_slam_teleop.py
  scripts/fastlio_mapping_visualizer.py
  scripts/classic_trotting_controller.py
  scripts/unpause_after_controllers.py
  scripts/open_mapping_entrance.py
  scripts/rl_mapping_speed_guard.py
  scripts/rl_takeover_stability_monitor.py
  scripts/goal_executor.py
  scripts/baseline_exploration_manager.py
  scripts/second_floor_exploration_manager.py
  scripts/stair_transition_manager.py
  scripts/stair_descent_manager.py
  scripts/waypoint_tour_manager.py
  scripts/baseline_telemetry.py
  scripts/truth_lidar_coverage_logger.py
  scripts/local_doorway_detector.py
  scripts/motion_consistency_guard.py
  scripts/ground_truth_odometry.py
  scripts/truth_registered_cloud.py
  scripts/truth_fastlio_status_relay.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Pure libraries imported by runtime nodes (same install dir for sys.path).
install(FILES
  scripts/frontier_information_structure.py
  scripts/motion_safety.py
  scripts/room0_mapping_core.py
  scripts/goal_executor_core.py
  scripts/baseline_planning_core.py
  scripts/lightweight_room_core.py
  scripts/scan_lite_path_refiner.py
  scripts/telemetry_core.py
  scripts/statistics_timing_report.py
  scripts/ufoexplorer_interface/corridor_semantic_core.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY docs
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
