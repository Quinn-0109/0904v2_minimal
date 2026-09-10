cmake_minimum_required(VERSION 3.0.2)
project(simenv_competitor)

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

add_executable(fastlio_voxel_mapper src/fastlio_voxel_mapper.cpp)
add_dependencies(fastlio_voxel_mapper ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_compile_features(fastlio_voxel_mapper PRIVATE cxx_std_14)
target_link_libraries(fastlio_voxel_mapper ${catkin_LIBRARIES} ${OCTOMAP_LIBRARIES})

add_executable(octomap_bt_inspector src/octomap_bt_inspector.cpp)
target_compile_features(octomap_bt_inspector PRIVATE cxx_std_14)
target_link_libraries(octomap_bt_inspector ${OCTOMAP_LIBRARIES})

add_executable(fuel_lite_planner src/fuel_lite_planner.cpp)
target_compile_features(fuel_lite_planner PRIVATE cxx_std_14)
target_link_libraries(fuel_lite_planner
  ${catkin_LIBRARIES} ${OCTOMAP_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(octomap_goal_validator src/octomap_goal_validator.cpp)
target_compile_features(octomap_goal_validator PRIVATE cxx_std_14)
target_link_libraries(octomap_goal_validator ${OCTOMAP_LIBRARIES})

add_executable(octomap_path_validator src/octomap_path_validator.cpp)
target_compile_features(octomap_path_validator PRIVATE cxx_std_14)
target_link_libraries(octomap_path_validator ${OCTOMAP_LIBRARIES})

add_executable(octomap_room_analyzer src/octomap_room_analyzer.cpp)
target_compile_features(octomap_room_analyzer PRIVATE cxx_std_14)
target_link_libraries(octomap_room_analyzer ${OCTOMAP_LIBRARIES})

add_executable(octomap_floor_projection_exporter
  src/octomap_floor_projection_exporter.cpp)
target_compile_features(octomap_floor_projection_exporter PRIVATE cxx_std_14)
target_link_libraries(octomap_floor_projection_exporter ${OCTOMAP_LIBRARIES})

add_executable(octomap_floor_filter src/octomap_floor_filter.cpp)
target_compile_features(octomap_floor_filter PRIVATE cxx_std_14)
target_link_libraries(octomap_floor_filter ${OCTOMAP_LIBRARIES})

install(TARGETS fastlio_voxel_mapper octomap_bt_inspector fuel_lite_planner
  octomap_goal_validator octomap_path_validator octomap_room_analyzer
  octomap_floor_projection_exporter octomap_floor_filter
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

catkin_install_python(PROGRAMS
  scripts/f1_stair_test_odom_bridge.py
  scripts/danger_detector.py
  scripts/red_ball_detector.py
  scripts/yolo_semantic_detector.py
  scripts/yolo_semantic_postprocess.py
  scripts/hazard_candidate_tracker.py
  scripts/visual_coverage_mapper.py
  scripts/room_visual_sweep_manager.py
  scripts/rgb_video_recorder.py
  scripts/xvfb_display_guard.py
  scripts/livox_slam_bridge.py
  scripts/oracle_tare_bridge.py
  scripts/oracle_room_tour.py
  scripts/radar_camera_attention.py
  scripts/realsense_tare_bridge.py
  scripts/compliant_tare_bridge.py
  scripts/cmd_vel_guard.py
  scripts/compliant_fall_recovery.py
  scripts/controller_mode_bootstrap.py
  scripts/lobby_egress_seed.py
  scripts/compliant_waypoint_follower.py
  scripts/structure_aware_explorer.py
  scripts/fastlio_pointcloud_adapter.py
  scripts/pointcloud2livox_livo.py
  scripts/trajectory_recorder.py
  scripts/localization_health_monitor.py
  scripts/localization_motion_benchmark.py
  scripts/speed_audit.py
  scripts/online_floor_mapper.py
  scripts/first_floor_mission_manager.py
  scripts/adaptive_speed_guard.py
  scripts/manual_slam_teleop.py
  scripts/fastlio_mapping_visualizer.py
  scripts/classic_trotting_controller.py
  scripts/unpause_after_controllers.py
  scripts/open_mapping_entrance.py
  scripts/room0_mapping_controller.py
  scripts/rl_mapping_speed_guard.py
  scripts/goal_executor.py
  scripts/single_exploration_goal.py
  scripts/goal_stability_monitor.py
  scripts/stair_entry_visual_servo.py
  scripts/stair_return_turn_entry.py
  scripts/stair_visual_depth_detector.py
  scripts/stair_entry_mission_adapter.py
  scripts/exploration_manager.py
  scripts/baseline_exploration_manager.py
  scripts/second_floor_exploration_manager.py
  scripts/stair_transition_manager.py
  scripts/stair_descent_manager.py
  scripts/isolated_descent_trigger.py
  scripts/stair_full_climb_driver.py
  scripts/elevator_transition_manager.py
  scripts/baseline_telemetry.py
  scripts/localization_truth_recovery.py
  scripts/truth_lidar_coverage_logger.py
  scripts/local_doorway_detector.py
  scripts/motion_consistency_guard.py
  scripts/structured_exploration_manager.py
  scripts/rl_takeover_stability_monitor.py
  scripts/frontier_cluster_node.py
  scripts/exploration_graph_node.py
  scripts/hierarchical_explorer_node.py
  scripts/compare_hierarchical_runs.py
  scripts/tare_interface.py
  scripts/tare_planner_diagnostics.py
  scripts/tare_goal_arbitrator.py
  scripts/tare_goal_refiner.py
  scripts/tare_fuel_hybrid_selector.py
  scripts/local_exploration_path_generator.py
  scripts/ufoexplorer_interface/ufo_map_bridge.py
  scripts/ufoexplorer_interface/ufo_corridor_semantic.py
  scripts/ufoexplorer_interface/ufo_room_semantic_scheduler.py
  scripts/ufoexplorer_interface/ufoexplorer_interface.py
  scripts/ufoexplorer_interface/ufo_path_adapter.py
  scripts/ufoexplorer_interface/ufoexplorer_status_monitor.py
  scripts/velocity_watchdog.py
  scripts/compare_explorer_backends.py
  scripts/far_interface.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Pure libraries imported by runtime nodes (same install dir for sys.path).
install(FILES
  scripts/frontier_information_structure.py
  scripts/detection_tracking.py
  scripts/motion_safety.py
  scripts/first_floor_core.py
  scripts/room0_mapping_core.py
  scripts/goal_executor_core.py
  scripts/exploration_manager_core.py
  scripts/baseline_planning_core.py
  scripts/lightweight_room_core.py
  scripts/scan_lite_path_refiner.py
  scripts/telemetry_core.py
  scripts/structured_topology_core.py
  scripts/hierarchical_ros_utils.py
  scripts/tare_goal_refinement_core.py
  scripts/tare_goal_arbitration_core.py
  scripts/tare_fuel_hybrid_core.py
  scripts/local_exploration_path_core.py
  scripts/ufoexplorer_interface/ufo_path_core.py
  scripts/ufoexplorer_interface/corridor_semantic_core.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY
  scripts/exploration_graph
  scripts/hierarchical_explorer
  scripts/planner_interface
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY docs
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(test/test_tare_fuel_hybrid.py)
  catkin_add_nosetests(test/test_tare_goal_arbitration.py)
  catkin_add_nosetests(test/test_local_exploration_path.py)
  catkin_add_nosetests(test/test_ufo_path_adapter.py)
endif()
