cmake_minimum_required(VERSION 3.0)
project(simenv_bridge)
find_package(catkin REQUIRED COMPONENTS rospy nav_msgs sensor_msgs geometry_msgs std_msgs)
catkin_package()
if(CATKIN_ENABLE_TESTING)
  find_package(rostest REQUIRED)
  add_rostest(test/mock_three_floor_ros_integration.test)
endif()
catkin_install_python(PROGRAMS
  scripts/odom_relay.py
  scripts/scan_to_map.py
  scripts/scan_to_uf.py
  scripts/export_uf_global_map.py
  scripts/export_livox_gt_map.py
  scripts/capture_livox_gt_sweep_map.py
  scripts/crop_sweep_map_to_layout.py
  scripts/auto_explore_trace_map.py
  scripts/cmd_vel_to_model_state.py
  scripts/grid_map.py
  scripts/map_saver.py
  scripts/waypoint_follower.py
  scripts/waypoint_follower_world.py
  scripts/layout_boundary_publisher.py
  scripts/tare_trace_map.py
  scripts/merge_trace_results.py
  scripts/auto_door.py
  scripts/floorplan.py
  scripts/body_cmd_vel_driver.py
  scripts/scanplanner_goal_sequencer.py
  scripts/scanplanner_record.py
  scripts/danger_detector.py
  scripts/gazebo_image_to_ros.py
  scripts/camera_video_recorder.py
  scripts/rl_floor_command_mux.py
  scripts/scanplanner_three_floor_goal_sequencer.py
  scripts/stair_transition_manager.py
  scripts/stair_descent_manager.py
  scripts/stair_descent_smoke_trigger.py
  scripts/check_stair_descent_smoke.py
  scripts/prepare_three_floor_scene.py
  scripts/randomize_three_floor_scene.py
  scripts/classic_trotting_controller.py
  scripts/devel_space_resolver.py
  scripts/three_floor_rl_mission_supervisor.py
  scripts/check_three_floor_rl_mission.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(PROGRAMS scripts/run_stair_descent_smoke.sh
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(DIRECTORY launch config vendor DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
install(FILES scripts/gazebo_image_to_ros.cpp
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/scripts)
