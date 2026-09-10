#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import sys
import types
import unittest


def _install_ros_stubs():
    sys.modules["rospy"] = types.ModuleType("rospy")
    geometry_msgs_msg = types.ModuleType("geometry_msgs.msg")
    geometry_msgs_msg.PointStamped = type("PointStamped", (), {})
    geometry_msgs_msg.Twist = type("Twist", (), {})
    geometry_msgs_msg.PoseStamped = type("PoseStamped", (), {})
    nav_msgs_msg = types.ModuleType("nav_msgs.msg")
    nav_msgs_msg.Odometry = type("Odometry", (), {})
    nav_msgs_msg.Path = type("Path", (), {})
    sys.modules["geometry_msgs"] = types.ModuleType("geometry_msgs")
    sys.modules["geometry_msgs.msg"] = geometry_msgs_msg
    sys.modules["nav_msgs"] = types.ModuleType("nav_msgs")
    sys.modules["nav_msgs.msg"] = nav_msgs_msg


def _load_module():
    _install_ros_stubs()
    script = Path(__file__).resolve().parents[1] / "scripts" / "waypoint_follower_world.py"
    spec = importlib.util.spec_from_file_location("waypoint_follower_world_under_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WaypointFollowerWorldTest(unittest.TestCase):
    def test_compute_world_cmd_points_toward_target(self):
        mod = _load_module()

        vx, vy, done = mod.compute_world_cmd(1.0, 2.0, 1.0, 5.0, 0.8, 0.2, 0.3)

        self.assertFalse(done)
        self.assertAlmostEqual(vx, 0.0, places=6)
        self.assertGreater(vy, 0.0)
        self.assertLessEqual((vx * vx + vy * vy) ** 0.5, 0.2 + 1e-6)

    def test_compute_world_cmd_stops_near_goal(self):
        mod = _load_module()

        vx, vy, done = mod.compute_world_cmd(1.0, 2.0, 1.1, 2.1, 0.8, 0.2, 0.3)

        self.assertTrue(done)
        self.assertEqual(vx, 0.0)
        self.assertEqual(vy, 0.0)

    def test_latched_waypoint_ignores_replans_until_goal_or_timeout(self):
        mod = _load_module()

        active = (0.0, 10.0, 0.0, 1.0)
        new_wp = (3.0, 5.0, 0.0, 2.0)

        kept, reached = mod.select_active_waypoint(
            active, new_wp, robot_x=0.0, robot_y=7.2, now=2.0,
            goal_tol=0.35, latch_timeout=20.0, latch_enabled=True,
        )

        self.assertEqual(kept, active)
        self.assertFalse(reached)

        replaced, reached = mod.select_active_waypoint(
            active, new_wp, robot_x=0.0, robot_y=9.9, now=3.0,
            goal_tol=0.35, latch_timeout=20.0, latch_enabled=True,
        )

        self.assertEqual(replaced, new_wp)
        self.assertTrue(reached)

    def test_latched_waypoint_expires(self):
        mod = _load_module()

        active = (0.0, 10.0, 0.0, 1.0)
        new_wp = (3.0, 5.0, 0.0, 25.0)

        selected, reached = mod.select_active_waypoint(
            active, new_wp, robot_x=0.0, robot_y=7.2, now=25.0,
            goal_tol=0.35, latch_timeout=20.0, latch_enabled=True,
        )

        self.assertEqual(selected, new_wp)
        self.assertFalse(reached)

    def test_active_latched_waypoint_blocks_path_preference_until_reached_or_timeout(self):
        mod = _load_module()

        active_wp = (5.4, 3.8, 0.6, 100.0)

        self.assertFalse(mod.path_preference_allowed(
            active_wp=active_wp,
            robot_x=-0.2,
            robot_y=3.3,
            now=105.0,
            goal_tol=0.35,
            latch_timeout=20.0,
            latch_enabled=True,
        ))
        self.assertTrue(mod.path_preference_allowed(
            active_wp=active_wp,
            robot_x=5.35,
            robot_y=3.78,
            now=106.0,
            goal_tol=0.35,
            latch_timeout=20.0,
            latch_enabled=True,
        ))
        self.assertTrue(mod.path_preference_allowed(
            active_wp=active_wp,
            robot_x=-0.2,
            robot_y=3.3,
            now=125.0,
            goal_tol=0.35,
            latch_timeout=20.0,
            latch_enabled=True,
        ))

    def test_path_target_uses_forward_lookahead_from_path_start_near_robot(self):
        mod = _load_module()
        # The return-home tail is spatially closer, but the path starts at the
        # current robot pose; execution should advance along the path, not jump
        # to the nearby tail and drive backward.
        path = [
            (0.0, 5.0, 0.0),
            (0.0, 6.0, 0.0),
            (0.0, 8.2, 0.0),
            (0.0, 10.0, 0.0),
            (0.5, 4.8, 0.0),
        ]

        target = mod.select_path_target(
            robot_x=0.0,
            robot_y=5.0,
            path=path,
            lookahead=2.0,
            anchor_tolerance=1.0,
        )

        self.assertEqual(target, (0.0, 8.2, 0.0))

    def test_path_target_falls_back_to_nearest_segment_when_path_start_is_stale(self):
        mod = _load_module()
        path = [
            (10.0, 10.0, 0.0),
            (1.0, 5.0, 0.0),
            (1.0, 6.0, 0.0),
            (1.0, 8.0, 0.0),
        ]

        target = mod.select_path_target(
            robot_x=1.0,
            robot_y=5.2,
            path=path,
            lookahead=1.5,
            anchor_tolerance=1.0,
        )

        self.assertEqual(target, (1.0, 8.0, 0.0))

    def test_path_target_returns_none_for_degenerate_path(self):
        mod = _load_module()
        path = [
            (2.3, 5.3, 0.0),
            (2.3, 5.3, 0.0),
            (2.3, 5.3, 0.0),
        ]

        target = mod.select_path_target(
            robot_x=2.3,
            robot_y=5.3,
            path=path,
            lookahead=2.0,
            anchor_tolerance=1.0,
            min_path_length=1.0,
        )

        self.assertIsNone(target)

    def test_explore_launch_can_use_world_frame_follower(self):
        launch = Path(__file__).resolve().parents[1] / "launch" / "explore_simenv.launch"
        text = launch.read_text()

        self.assertIn('name="follow_mode"', text)
        self.assertIn('type="waypoint_follower_world.py"', text)
        self.assertIn("""if="$(eval arg('follow') and arg('follow_mode') == 'world')\"""", text)

    def test_explore_launch_defaults_to_tare_waypoint_following(self):
        launch = Path(__file__).resolve().parents[1] / "launch" / "explore_simenv.launch"
        text = launch.read_text()

        self.assertIn('<param name="prefer_path"  value="true"/>', text)

    def test_explore_launch_latches_tare_waypoints_to_avoid_replan_oscillation(self):
        launch = Path(__file__).resolve().parents[1] / "launch" / "explore_simenv.launch"
        text = launch.read_text()

        self.assertIn('<param name="latch_waypoint" value="true"/>', text)

    def test_explore_launch_uses_driver_matched_world_follower_speed(self):
        launch = Path(__file__).resolve().parents[1] / "launch" / "explore_simenv.launch"
        text = launch.read_text()

        self.assertIn('<param name="v_max"        value="1.0"/>', text)

    def test_explore_launch_starts_world_cmd_vel_driver_at_one_meter_per_second(self):
        launch = Path(__file__).resolve().parents[1] / "launch" / "explore_simenv.launch"
        text = launch.read_text()

        self.assertIn('type="cmd_vel_to_model_state.py"', text)
        self.assertIn('<param name="max_vx" value="1.0"/>', text)
        self.assertIn('<param name="max_vy" value="1.0"/>', text)
        self.assertIn('<param name="lock_z" value="0.6"/>', text)

    def test_explore_launch_uses_one_meter_per_second_tf_follower_speed(self):
        launch = Path(__file__).resolve().parents[1] / "launch" / "explore_simenv.launch"
        text = launch.read_text()

        self.assertNotIn('<param name="v_max"        value="0.10"/>', text)

    def test_world_follower_default_speed_is_one_meter_per_second(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "waypoint_follower_world.py"
        text = script.read_text()

        self.assertIn('rospy.get_param("~v_max", 1.0)', text)

    def test_manual_sim_forward_and_back_default_to_one_meter_per_second(self):
        script = Path(__file__).resolve().parents[3] / "docker" / "sim.sh"
        text = script.read_text()

        self.assertIn('fwd  [1.0]', text)
        self.assertIn('back [1.0]', text)
        self.assertIn('publish_vel "${2:-1.0}" 0 0', text)
        self.assertIn('publish_vel "${2:-1.0}" 0 0_neg', text)

    def test_run_tare_explore_does_not_kill_external_trace_recorder(self):
        script = Path(__file__).resolve().parents[3] / "docker" / "run_tare.sh"
        text = script.read_text()
        stop_body = text.split("stop_explore_nodes() {", 1)[1].split("restart_livox_bridge()", 1)[0]

        self.assertNotIn("/tare_trace_map", stop_body)

    def test_path_target_advances_to_global_path_goal_when_start_near_robot(self):
        mod = _load_module()
        path = [
            (-3.2, 6.4, 0.6),
            (-0.4, 11.6, 0.6),
            (10.4, 2.0, 0.6),
            (-3.2, 6.4, 0.6),
        ]

        target = mod.select_path_target(
            robot_x=-3.1,
            robot_y=6.5,
            path=path,
            lookahead=2.0,
            anchor_tolerance=3.0,
        )

        self.assertEqual(target, (-0.4, 11.6, 0.6))

    def test_topology_guard_routes_corridor_to_room_door_then_inside_room(self):
        mod = _load_module()
        metadata = {
            "floors": [{
                "corridor_bounds": {"x_min": -1.1, "x_max": 1.1, "y_min": 7.85, "y_max": 35.91},
                "rooms": [{
                    "id": "floor_0_room_1",
                    "bounds": {"x_min": 1.1, "x_max": 9.5, "y_min": 7.85, "y_max": 35.91},
                    "door_pose": [1.1, 21.88, 1.2, 0.0, 0.0, 3.14],
                }],
            }]
        }
        topology = mod.parse_layout_topology(metadata)

        target, entered, overridden = mod.apply_topology_guard(
            robot_x=0.0,
            robot_y=10.0,
            target=(9.0, 6.0, 0.0),
            topology=topology,
            room_entered=False,
            door_tolerance=0.75,
            corridor_margin=0.5,
            room_entry_depth=2.0,
        )

        self.assertEqual(target, (0.0, 21.88, 0.0))
        self.assertFalse(entered)
        self.assertTrue(overridden)

        target, entered, overridden = mod.apply_topology_guard(
            robot_x=0.0,
            robot_y=21.7,
            target=(9.0, 6.0, 0.0),
            topology=topology,
            room_entered=False,
            door_tolerance=0.75,
            corridor_margin=0.5,
            room_entry_depth=2.0,
        )

        self.assertEqual(target, (3.1, 21.88, 0.0))
        self.assertFalse(entered)
        self.assertTrue(overridden)

    def test_topology_guard_routes_lobby_to_corridor_when_global_path_needs_corridor(self):
        mod = _load_module()
        metadata = {
            "floors": [{
                "lobby_bounds": {"x_min": -10.0, "x_max": 10.0, "y_min": 0.0, "y_max": 7.85},
                "corridor_bounds": {"x_min": -1.1, "x_max": 1.1, "y_min": 7.85, "y_max": 35.91},
                "rooms": [{
                    "id": "floor_0_room_1",
                    "bounds": {"x_min": 1.1, "x_max": 9.5, "y_min": 7.85, "y_max": 35.91},
                    "door_pose": [1.1, 21.88, 1.2, 0.0, 0.0, 3.14],
                }],
            }]
        }
        topology = mod.parse_layout_topology(metadata)
        global_path = [
            (0.0, 3.0, 0.6),
            (0.7, 12.2, 0.6),
            (9.1, 12.2, 0.6),
        ]

        target, entered, overridden = mod.apply_topology_guard(
            robot_x=0.2,
            robot_y=3.1,
            target=(-4.0, 4.9, 0.6),
            topology=topology,
            room_entered=False,
            door_tolerance=0.75,
            corridor_margin=0.5,
            room_entry_depth=2.0,
            global_path=global_path,
        )

        self.assertEqual(target, (0.0, 9.85, 0.6))
        self.assertFalse(entered)
        self.assertTrue(overridden)

    def test_lobby_bootstrap_targets_cover_lobby_width_before_corridor_entry(self):
        mod = _load_module()
        metadata = {
            "floors": [{
                "lobby_bounds": {"x_min": -10.0, "x_max": 10.0, "y_min": 0.0, "y_max": 7.85},
                "corridor_bounds": {"x_min": -1.1, "x_max": 1.1, "y_min": 7.85, "y_max": 35.91},
                "rooms": [{
                    "id": "floor_0_room_1",
                    "bounds": {"x_min": 1.1, "x_max": 9.5, "y_min": 7.85, "y_max": 35.91},
                    "door_pose": [1.1, 21.88, 1.2, 0.0, 0.0, 3.14],
                }],
            }]
        }
        topology = mod.parse_layout_topology(metadata)

        targets = mod.build_lobby_bootstrap_targets(topology, room_entry_depth=2.0, z=0.6)

        self.assertEqual(targets[0], (-8.0, 6.35, 0.6))
        self.assertEqual(targets[1], (8.0, 6.35, 0.6))
        self.assertEqual(targets[-1], (0.0, 9.85, 0.6))

    def test_lobby_bootstrap_intercepts_corridor_departure_once(self):
        mod = _load_module()
        metadata = {
            "floors": [{
                "lobby_bounds": {"x_min": -10.0, "x_max": 10.0, "y_min": 0.0, "y_max": 7.85},
                "corridor_bounds": {"x_min": -1.1, "x_max": 1.1, "y_min": 7.85, "y_max": 35.91},
                "rooms": [{
                    "id": "floor_0_room_1",
                    "bounds": {"x_min": 1.1, "x_max": 9.5, "y_min": 7.85, "y_max": 35.91},
                    "door_pose": [1.1, 21.88, 1.2, 0.0, 0.0, 3.14],
                }],
            }]
        }
        topology = mod.parse_layout_topology(metadata)
        state = {"active": False, "done": False, "index": 0}
        global_path = [(0.0, 3.0, 0.6), (0.0, 12.0, 0.6)]

        target, overridden = mod.select_lobby_bootstrap_target(
            robot_x=0.0,
            robot_y=3.0,
            target=(0.0, 9.85, 0.6),
            topology=topology,
            state=state,
            global_path=global_path,
            room_entry_depth=2.0,
            goal_tol=0.35,
        )

        self.assertEqual(target, (-8.0, 6.35, 0.6))
        self.assertTrue(overridden)
        self.assertTrue(state["active"])

    def test_lobby_bootstrap_starts_from_lobby_before_corridor_intent(self):
        mod = _load_module()
        metadata = {
            "floors": [{
                "lobby_bounds": {"x_min": -10.0, "x_max": 10.0, "y_min": 0.0, "y_max": 7.85},
                "corridor_bounds": {"x_min": -1.1, "x_max": 1.1, "y_min": 7.85, "y_max": 35.91},
                "rooms": [{
                    "id": "floor_0_room_1",
                    "bounds": {"x_min": 1.1, "x_max": 9.5, "y_min": 7.85, "y_max": 35.91},
                    "door_pose": [1.1, 21.88, 1.2, 0.0, 0.0, 3.14],
                }],
            }]
        }
        topology = mod.parse_layout_topology(metadata)
        state = {"active": False, "done": False, "index": 0}
        global_path = [(0.0, 3.0, 0.6), (1.0, 5.0, 0.6)]

        target, overridden = mod.select_lobby_bootstrap_target(
            robot_x=0.0,
            robot_y=2.0,
            target=(1.0, 5.0, 0.6),
            topology=topology,
            state=state,
            global_path=global_path,
            room_entry_depth=2.0,
            goal_tol=0.35,
        )

        self.assertEqual(target, (-8.0, 6.35, 0.6))
        self.assertTrue(overridden)

    def test_topology_guard_preserves_same_room_target_after_room_entry(self):
        mod = _load_module()
        metadata = {
            "floors": [{
                "corridor_bounds": {"x_min": -1.1, "x_max": 1.1, "y_min": 7.85, "y_max": 35.91},
                "rooms": [{
                    "id": "floor_0_room_1",
                    "bounds": {"x_min": 1.1, "x_max": 9.5, "y_min": 7.85, "y_max": 35.91},
                    "door_pose": [1.1, 21.88, 1.2, 0.0, 0.0, 3.14],
                }],
            }]
        }
        topology = mod.parse_layout_topology(metadata)

        target, entered, overridden = mod.apply_topology_guard(
            robot_x=3.0,
            robot_y=21.88,
            target=(8.0, 25.0, 0.0),
            topology=topology,
            room_entered=False,
            door_tolerance=0.75,
            corridor_margin=0.5,
            room_entry_depth=2.0,
        )

        self.assertEqual(target, (8.0, 25.0, 0.0))
        self.assertTrue(entered)
        self.assertFalse(overridden)

    def test_topology_guard_routes_to_target_room_door_after_previous_room_entry(self):
        mod = _load_module()
        metadata = {
            "floors": [{
                "corridor_bounds": {"x_min": -1.1, "x_max": 1.1, "y_min": 7.85, "y_max": 35.91},
                "rooms": [
                    {
                        "id": "floor_0_room_0",
                        "bounds": {"x_min": -9.5, "x_max": -1.1, "y_min": 7.85, "y_max": 35.91},
                        "door_pose": [-1.1, 21.88, 1.2, 0.0, 0.0, 0.0],
                    },
                    {
                        "id": "floor_0_room_1",
                        "bounds": {"x_min": 1.1, "x_max": 9.5, "y_min": 7.85, "y_max": 35.91},
                        "door_pose": [1.1, 21.88, 1.2, 0.0, 0.0, 3.14],
                    },
                ],
            }]
        }
        topology = mod.parse_layout_topology(metadata)

        target, entered, overridden = mod.apply_topology_guard(
            robot_x=-0.85,
            robot_y=16.9,
            target=(-4.36, 16.98, 0.0),
            topology=topology,
            room_entered=True,
            door_tolerance=0.75,
            corridor_margin=0.5,
            room_entry_depth=2.0,
        )

        self.assertEqual(target, (0.0, 21.88, 0.0))
        self.assertFalse(entered)
        self.assertTrue(overridden)

        target, entered, overridden = mod.apply_topology_guard(
            robot_x=-0.85,
            robot_y=21.7,
            target=(-4.36, 16.98, 0.0),
            topology=topology,
            room_entered=True,
            door_tolerance=0.75,
            corridor_margin=0.5,
            room_entry_depth=2.0,
        )

        self.assertEqual(target, (-3.1, 21.88, 0.0))
        self.assertFalse(entered)
        self.assertTrue(overridden)

    def test_topology_guard_preserves_deep_room_target_after_crossing_door(self):
        mod = _load_module()
        metadata = {
            "floors": [{
                "corridor_bounds": {"x_min": -1.1, "x_max": 1.1, "y_min": 7.85, "y_max": 35.91},
                "rooms": [{
                    "id": "floor_0_room_1",
                    "bounds": {"x_min": 1.1, "x_max": 9.5, "y_min": 7.85, "y_max": 35.91},
                    "door_pose": [1.1, 21.88, 1.2, 0.0, 0.0, 3.14],
                }],
            }]
        }
        topology = mod.parse_layout_topology(metadata)

        target, entered, overridden = mod.apply_topology_guard(
            robot_x=1.2,
            robot_y=21.88,
            target=(8.7, 24.8, 0.0),
            topology=topology,
            room_entered=False,
            door_tolerance=0.75,
            corridor_margin=0.5,
            room_entry_depth=2.0,
        )

        self.assertEqual(target, (8.7, 24.8, 0.0))
        self.assertTrue(entered)
        self.assertFalse(overridden)

    def test_room_entry_commit_ignores_transient_replan_until_robot_reaches_entry_target(self):
        mod = _load_module()
        metadata = {
            "floors": [{
                "corridor_bounds": {"x_min": -1.1, "x_max": 1.1, "y_min": 7.85, "y_max": 35.91},
                "rooms": [
                    {
                        "id": "floor_0_room_0",
                        "bounds": {"x_min": -9.5, "x_max": -1.1, "y_min": 7.85, "y_max": 35.91},
                        "door_pose": [-1.1, 21.88, 1.2, 0.0, 0.0, 0.0],
                    },
                    {
                        "id": "floor_0_room_1",
                        "bounds": {"x_min": 1.1, "x_max": 9.5, "y_min": 7.85, "y_max": 35.91},
                        "door_pose": [1.1, 21.88, 1.2, 0.0, 0.0, 3.14],
                    },
                ],
            }]
        }
        topology = mod.parse_layout_topology(metadata)

        commit = None
        target, commit, overridden = mod.apply_room_entry_commit(
            robot_x=0.8,
            robot_y=21.7,
            target=(5.4, 20.5, 0.0),
            topology=topology,
            commit=commit,
            door_tolerance=0.75,
            room_entry_depth=2.0,
        )

        self.assertEqual(target, (3.1, 21.88, 0.0))
        self.assertEqual(commit["room_id"], "floor_0_room_1")
        self.assertTrue(overridden)

        target, commit, overridden = mod.apply_room_entry_commit(
            robot_x=0.9,
            robot_y=21.75,
            target=(-6.5, 21.7, 0.0),
            topology=topology,
            commit=commit,
            door_tolerance=0.75,
            room_entry_depth=2.0,
        )

        self.assertEqual(target, (3.1, 21.88, 0.0))
        self.assertEqual(commit["room_id"], "floor_0_room_1")
        self.assertTrue(overridden)

        target, commit, overridden = mod.apply_room_entry_commit(
            robot_x=1.25,
            robot_y=21.85,
            target=(-6.5, 21.7, 0.0),
            topology=topology,
            commit=commit,
            door_tolerance=0.75,
            room_entry_depth=2.0,
        )

        self.assertEqual(target, (3.1, 21.88, 0.0))
        self.assertEqual(commit["room_id"], "floor_0_room_1")
        self.assertTrue(overridden)

        target, commit, overridden = mod.apply_room_entry_commit(
            robot_x=3.0,
            robot_y=21.88,
            target=(-6.5, 21.7, 0.0),
            topology=topology,
            commit=commit,
            door_tolerance=0.75,
            room_entry_depth=2.0,
        )

        self.assertEqual(target, (-6.5, 21.7, 0.0))
        self.assertIsNone(commit)
        self.assertFalse(overridden)

    def test_room_entry_commit_locks_target_room_before_reaching_door(self):
        mod = _load_module()
        metadata = {
            "floors": [{
                "lobby_bounds": {"x_min": -10.0, "x_max": 10.0, "y_min": 0.0, "y_max": 7.85},
                "corridor_bounds": {"x_min": -1.1, "x_max": 1.1, "y_min": 7.85, "y_max": 35.91},
                "rooms": [
                    {
                        "id": "floor_0_room_0",
                        "bounds": {"x_min": -9.5, "x_max": -1.1, "y_min": 7.85, "y_max": 35.91},
                        "door_pose": [-1.1, 21.88, 1.2, 0.0, 0.0, 0.0],
                    },
                    {
                        "id": "floor_0_room_1",
                        "bounds": {"x_min": 1.1, "x_max": 9.5, "y_min": 7.85, "y_max": 35.91},
                        "door_pose": [1.1, 21.88, 1.2, 0.0, 0.0, 3.14],
                    },
                ],
            }]
        }
        topology = mod.parse_layout_topology(metadata)

        commit = None
        target, commit, overridden = mod.apply_room_entry_commit(
            robot_x=0.0,
            robot_y=18.9,
            target=(1.87, 20.64, 0.0),
            topology=topology,
            commit=commit,
            door_tolerance=0.75,
            room_entry_depth=2.0,
        )

        self.assertEqual(target, (0.0, 21.88, 0.0))
        self.assertEqual(commit["room_id"], "floor_0_room_1")
        self.assertTrue(overridden)

        target, commit, overridden = mod.apply_room_entry_commit(
            robot_x=0.0,
            robot_y=21.2,
            target=(-6.5, 21.7, 0.0),
            topology=topology,
            commit=commit,
            door_tolerance=0.75,
            room_entry_depth=2.0,
        )

        self.assertEqual(target, (3.1, 21.88, 0.0))
        self.assertEqual(commit["room_id"], "floor_0_room_1")
        self.assertTrue(overridden)

    def test_routing_commits_to_room_before_guard_rewrites_target_to_door(self):
        mod = _load_module()
        metadata = {
            "floors": [{
                "corridor_bounds": {"x_min": -1.1, "x_max": 1.1, "y_min": 7.85, "y_max": 35.91},
                "rooms": [
                    {
                        "id": "floor_0_room_0",
                        "bounds": {"x_min": -9.5, "x_max": -1.1, "y_min": 7.85, "y_max": 35.91},
                        "door_pose": [-1.1, 21.88, 1.2, 0.0, 0.0, 0.0],
                    },
                    {
                        "id": "floor_0_room_1",
                        "bounds": {"x_min": 1.1, "x_max": 9.5, "y_min": 7.85, "y_max": 35.91},
                        "door_pose": [1.1, 21.88, 1.2, 0.0, 0.0, 3.14],
                    },
                ],
            }]
        }
        topology = mod.parse_layout_topology(metadata)

        target, room_entered, commit, overridden = mod.route_topology_target(
            robot_x=-1.5,
            robot_y=19.35,
            target=(4.22, 26.53, 0.0),
            topology=topology,
            room_entered=False,
            commit=None,
            door_tolerance=0.75,
            corridor_margin=0.5,
            room_entry_depth=2.0,
        )

        self.assertEqual(target, (0.0, 21.88, 0.0))
        self.assertTrue(room_entered)
        self.assertEqual(commit["room_id"], "floor_0_room_1")
        self.assertTrue(overridden)

    def test_topology_guard_does_not_guess_left_room_for_corridor_waypoint(self):
        mod = _load_module()
        metadata = {
            "floors": [{
                "corridor_bounds": {"x_min": -1.1, "x_max": 1.1, "y_min": 7.85, "y_max": 35.91},
                "rooms": [
                    {
                        "id": "floor_0_room_0",
                        "bounds": {"x_min": -9.5, "x_max": -1.1, "y_min": 7.85, "y_max": 35.91},
                        "door_pose": [-1.1, 21.88, 1.2, 0.0, 0.0, 0.0],
                    },
                    {
                        "id": "floor_0_room_1",
                        "bounds": {"x_min": 1.1, "x_max": 9.5, "y_min": 7.85, "y_max": 35.91},
                        "door_pose": [1.1, 21.88, 1.2, 0.0, 0.0, 3.14],
                    },
                ],
            }]
        }
        topology = mod.parse_layout_topology(metadata)

        target, entered, overridden = mod.apply_topology_guard(
            robot_x=0.0,
            robot_y=20.9,
            target=(-0.53, 25.44, 0.0),
            topology=topology,
            room_entered=False,
            door_tolerance=0.75,
            corridor_margin=0.5,
            room_entry_depth=2.0,
        )

        self.assertEqual(target, (-0.53, 25.44, 0.0))
        self.assertFalse(entered)
        self.assertFalse(overridden)

    def test_topology_guard_allows_corridor_to_lobby_waypoint(self):
        mod = _load_module()
        metadata = {
            "floors": [{
                "lobby_bounds": {"x_min": -10.0, "x_max": 10.0, "y_min": 0.0, "y_max": 7.85},
                "corridor_bounds": {"x_min": -1.1, "x_max": 1.1, "y_min": 7.85, "y_max": 35.91},
                "rooms": [
                    {
                        "id": "floor_0_room_0",
                        "bounds": {"x_min": -9.5, "x_max": -1.1, "y_min": 7.85, "y_max": 21.88},
                        "door_pose": [-1.1, 14.865, 1.2, 0.0, 0.0, 0.0],
                    },
                    {
                        "id": "floor_0_room_1",
                        "bounds": {"x_min": 1.1, "x_max": 9.5, "y_min": 7.85, "y_max": 21.88},
                        "door_pose": [1.1, 14.865, 1.2, 0.0, 0.0, 3.14],
                    },
                ],
            }]
        }
        topology = mod.parse_layout_topology(metadata)

        target, entered, overridden = mod.apply_topology_guard(
            robot_x=-1.0,
            robot_y=14.5,
            target=(-0.53, 2.62, 0.0),
            topology=topology,
            room_entered=False,
            door_tolerance=0.75,
            corridor_margin=0.5,
            room_entry_depth=2.0,
        )

        self.assertEqual(target, (-0.53, 2.62, 0.0))
        self.assertFalse(entered)
        self.assertFalse(overridden)

    def test_path_following_uses_latest_waypoint_as_room_intent_when_path_target_is_corridor(self):
        mod = _load_module()
        metadata = {
            "floors": [{
                "corridor_bounds": {"x_min": -1.1, "x_max": 1.1, "y_min": 7.85, "y_max": 35.91},
                "rooms": [
                    {
                        "id": "floor_0_room_0",
                        "bounds": {"x_min": -9.5, "x_max": -1.1, "y_min": 7.85, "y_max": 35.91},
                        "door_pose": [-1.1, 21.88, 1.2, 0.0, 0.0, 0.0],
                    },
                    {
                        "id": "floor_0_room_1",
                        "bounds": {"x_min": 1.1, "x_max": 9.5, "y_min": 7.85, "y_max": 35.91},
                        "door_pose": [1.1, 21.88, 1.2, 0.0, 0.0, 3.14],
                    },
                ],
            }]
        }
        topology = mod.parse_layout_topology(metadata)

        intent = mod.select_topology_intent_target(
            path_target=(0.6, 22.99, 0.0),
            latest_wp=(6.6, 12.19, 0.0, 10.0),
            latest_is_fresh=True,
            topology=topology,
        )

        self.assertEqual(intent, (6.6, 12.19, 0.0, 10.0))

    def test_path_following_uses_latest_waypoint_as_room_intent_when_path_target_is_other_room(self):
        mod = _load_module()
        metadata = {
            "floors": [{
                "corridor_bounds": {"x_min": -1.1, "x_max": 1.1, "y_min": 7.85, "y_max": 35.91},
                "rooms": [
                    {
                        "id": "floor_0_room_0",
                        "bounds": {"x_min": -9.5, "x_max": -1.1, "y_min": 7.85, "y_max": 35.91},
                        "door_pose": [-1.1, 21.88, 1.2, 0.0, 0.0, 0.0],
                    },
                    {
                        "id": "floor_0_room_1",
                        "bounds": {"x_min": 1.1, "x_max": 9.5, "y_min": 7.85, "y_max": 35.91},
                        "door_pose": [1.1, 21.88, 1.2, 0.0, 0.0, 3.14],
                    },
                ],
            }]
        }
        topology = mod.parse_layout_topology(metadata)

        intent = mod.select_topology_intent_target(
            path_target=(-4.2, 21.83, 0.0),
            latest_wp=(1.8, 30.23, 0.0, 10.0),
            latest_is_fresh=True,
            topology=topology,
        )

        self.assertEqual(intent, (1.8, 30.23, 0.0, 10.0))

    def test_topology_guard_routes_room_exit_through_current_room_door(self):
        mod = _load_module()
        metadata = {
            "floors": [{
                "corridor_bounds": {"x_min": -1.1, "x_max": 1.1, "y_min": 7.85, "y_max": 35.91},
                "rooms": [
                    {
                        "id": "floor_0_room_0",
                        "bounds": {"x_min": -9.5, "x_max": -1.1, "y_min": 7.85, "y_max": 35.91},
                        "door_pose": [-1.1, 21.88, 1.2, 0.0, 0.0, 0.0],
                    },
                    {
                        "id": "floor_0_room_1",
                        "bounds": {"x_min": 1.1, "x_max": 9.5, "y_min": 7.85, "y_max": 35.91},
                        "door_pose": [1.1, 21.88, 1.2, 0.0, 0.0, 3.14],
                    },
                ],
            }]
        }
        topology = mod.parse_layout_topology(metadata)

        target, room_entered, overridden = mod.apply_topology_guard(
            robot_x=5.2,
            robot_y=24.8,
            target=(-3.8, 26.5, 0.0),
            topology=topology,
            room_entered=True,
            door_tolerance=0.75,
            corridor_margin=0.5,
            room_entry_depth=2.0,
        )

        self.assertEqual(target, (0.0, 21.88, 0.0))
        self.assertTrue(room_entered)
        self.assertTrue(overridden)

        target, room_entered, overridden = mod.apply_topology_guard(
            robot_x=1.15,
            robot_y=22.35,
            target=(0.0, 22.85, 0.0),
            topology=topology,
            room_entered=True,
            door_tolerance=0.75,
            corridor_margin=0.5,
            room_entry_depth=2.0,
        )

        self.assertEqual(target, (0.0, 21.88, 0.0))
        self.assertTrue(room_entered)
        self.assertTrue(overridden)

        target, room_entered, overridden = mod.apply_topology_guard(
            robot_x=1.05,
            robot_y=21.9,
            target=(-3.8, 26.5, 0.0),
            topology=topology,
            room_entered=True,
            door_tolerance=0.75,
            corridor_margin=0.5,
            room_entry_depth=2.0,
        )

        self.assertEqual(target, (-3.1, 21.88, 0.0))
        self.assertFalse(room_entered)
        self.assertTrue(overridden)

    def test_topology_guard_does_not_exit_room_while_current_target_is_inside_same_room(self):
        mod = _load_module()
        metadata = {
            "floors": [{
                "corridor_bounds": {"x_min": -1.1, "x_max": 1.1, "y_min": 7.85, "y_max": 35.91},
                "rooms": [
                    {
                        "id": "floor_0_room_0",
                        "bounds": {"x_min": -9.5, "x_max": -1.1, "y_min": 7.85, "y_max": 35.91},
                        "door_pose": [-1.1, 21.88, 1.2, 0.0, 0.0, 0.0],
                    },
                    {
                        "id": "floor_0_room_1",
                        "bounds": {"x_min": 1.1, "x_max": 9.5, "y_min": 7.85, "y_max": 35.91},
                        "door_pose": [1.1, 21.88, 1.2, 0.0, 0.0, 3.14],
                    },
                ],
            }]
        }
        topology = mod.parse_layout_topology(metadata)
        global_path = [
            (6.9, 31.9, 0.6),
            (7.5, 32.5, 0.6),
            (-0.9, 34.9, 0.6),
            (-0.9, 21.7, 0.6),
        ]

        target, room_entered, overridden = mod.apply_topology_guard(
            robot_x=7.2,
            robot_y=32.1,
            target=(7.5, 32.5, 0.6),
            topology=topology,
            room_entered=True,
            door_tolerance=0.75,
            corridor_margin=0.5,
            room_entry_depth=2.0,
            global_path=global_path,
        )

        self.assertEqual(target, (7.5, 32.5, 0.6))
        self.assertTrue(room_entered)
        self.assertFalse(overridden)

    def test_route_topology_target_clamps_slightly_outside_upper_room_before_routing(self):
        mod = _load_module()
        metadata = {
            "floors": [{
                "lobby_bounds": {"x_min": -10.0, "x_max": 10.0, "y_min": 0.0, "y_max": 7.85},
                "corridor_bounds": {"x_min": -1.1, "x_max": 1.1, "y_min": 7.85, "y_max": 35.91},
                "rooms": [
                    {
                        "id": "floor_0_room_0",
                        "bounds": {"x_min": -9.5, "x_max": -1.1, "y_min": 7.85, "y_max": 21.88},
                        "door_pose": [-1.1, 14.865, 1.2, 0.0, 0.0, 0.0],
                    },
                    {
                        "id": "floor_0_room_1",
                        "bounds": {"x_min": 1.1, "x_max": 9.5, "y_min": 7.85, "y_max": 21.88},
                        "door_pose": [1.1, 14.865, 1.2, 0.0, 0.0, 3.14],
                    },
                    {
                        "id": "floor_0_room_2",
                        "bounds": {"x_min": -9.5, "x_max": -1.1, "y_min": 21.88, "y_max": 35.91},
                        "door_pose": [-1.1, 28.895, 1.2, 0.0, 0.0, 0.0],
                    },
                    {
                        "id": "floor_0_room_3",
                        "bounds": {"x_min": 1.1, "x_max": 9.5, "y_min": 21.88, "y_max": 35.91},
                        "door_pose": [1.1, 28.895, 1.2, 0.0, 0.0, 3.14],
                    },
                ],
            }]
        }
        topology = mod.parse_layout_topology(metadata)

        target, room_entered, commit, overridden = mod.route_topology_target(
            robot_x=0.4,
            robot_y=14.6,
            target=(10.2, 22.0, 0.6),
            topology=topology,
            room_entered=False,
            commit=None,
            door_tolerance=0.75,
            corridor_margin=0.5,
            room_entry_depth=2.0,
        )

        self.assertEqual(target, (0.0, 28.895, 0.0))
        self.assertFalse(room_entered)
        self.assertEqual(commit, {"room_id": "floor_0_room_3"})
        self.assertTrue(overridden)

    def test_topology_latch_replaces_active_waypoint_when_new_waypoint_enters_different_room(self):
        mod = _load_module()
        metadata = {
            "floors": [{
                "lobby_bounds": {"x_min": -10.0, "x_max": 10.0, "y_min": 0.0, "y_max": 7.85},
                "corridor_bounds": {"x_min": -1.1, "x_max": 1.1, "y_min": 7.85, "y_max": 35.91},
                "rooms": [
                    {
                        "id": "floor_0_room_1",
                        "bounds": {"x_min": 1.1, "x_max": 9.5, "y_min": 7.85, "y_max": 21.88},
                        "door_pose": [1.1, 14.865, 1.2, 0.0, 0.0, 3.14],
                    },
                    {
                        "id": "floor_0_room_3",
                        "bounds": {"x_min": 1.1, "x_max": 9.5, "y_min": 21.88, "y_max": 35.91},
                        "door_pose": [1.1, 28.895, 1.2, 0.0, 0.0, 3.14],
                    },
                ],
            }]
        }
        topology = mod.parse_layout_topology(metadata)

        self.assertTrue(mod.should_replace_latched_waypoint(
            active_wp=(4.2, 8.8, 0.6, 100.0),
            new_wp=(4.2, 23.2, 0.6, 101.0),
            topology=topology,
        ))

    def test_topology_latch_does_not_replace_for_lobby_corridor_adjustments(self):
        mod = _load_module()
        metadata = {
            "floors": [{
                "lobby_bounds": {"x_min": -10.0, "x_max": 10.0, "y_min": 0.0, "y_max": 7.85},
                "corridor_bounds": {"x_min": -1.1, "x_max": 1.1, "y_min": 7.85, "y_max": 35.91},
                "rooms": [{
                    "id": "floor_0_room_0",
                    "bounds": {"x_min": -9.5, "x_max": -1.1, "y_min": 7.85, "y_max": 21.88},
                    "door_pose": [-1.1, 14.865, 1.2, 0.0, 0.0, 0.0],
                }],
            }]
        }
        topology = mod.parse_layout_topology(metadata)

        self.assertFalse(mod.should_replace_latched_waypoint(
            active_wp=(-0.6, 1.6, 0.6, 100.0),
            new_wp=(0.6, 8.4, 0.6, 101.0),
            topology=topology,
        ))

    def test_clamp_target_to_layout_keeps_lobby_target_inside_building(self):
        mod = _load_module()
        metadata = {
            "floors": [{
                "lobby_bounds": {"x_min": -10.0, "x_max": 10.0, "y_min": 0.0, "y_max": 7.85},
                "corridor_bounds": {"x_min": -1.1, "x_max": 1.1, "y_min": 7.85, "y_max": 35.91},
                "rooms": [{
                    "id": "floor_0_room_0",
                    "bounds": {"x_min": -9.5, "x_max": -1.1, "y_min": 7.85, "y_max": 35.91},
                    "door_pose": [-1.1, 21.88, 1.2, 0.0, 0.0, 0.0],
                }],
            }]
        }
        topology = mod.parse_layout_topology(metadata)

        clamped, changed = mod.clamp_target_to_layout((-7.76, -0.30, 0.35), topology, inset=0.05)

        self.assertEqual(clamped, (-7.76, 0.05, 0.35))
        self.assertTrue(changed)

    def test_clamp_target_to_layout_preserves_latched_waypoint_timestamp(self):
        mod = _load_module()
        metadata = {
            "floors": [{
                "lobby_bounds": {"x_min": -10.0, "x_max": 10.0, "y_min": 0.0, "y_max": 7.85},
                "corridor_bounds": {"x_min": -1.1, "x_max": 1.1, "y_min": 7.85, "y_max": 35.91},
                "rooms": [{
                    "id": "floor_0_room_0",
                    "bounds": {"x_min": -9.5, "x_max": -1.1, "y_min": 7.85, "y_max": 35.91},
                    "door_pose": [-1.1, 21.88, 1.2, 0.0, 0.0, 0.0],
                }],
            }]
        }
        topology = mod.parse_layout_topology(metadata)

        clamped, changed = mod.clamp_target_to_layout((-7.76, -0.30, 0.35, 123.0), topology, inset=0.05)

        self.assertEqual(clamped, (-7.76, 0.05, 0.35, 123.0))
        self.assertTrue(changed)


if __name__ == "__main__":
    unittest.main()
