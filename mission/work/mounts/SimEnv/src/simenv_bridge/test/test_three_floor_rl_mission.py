#!/usr/bin/env python3
import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "three_floor_rl_mission.json"


def load_file_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_checker():
    return load_file_module(
        "check_three_floor_rl_mission_under_test",
        ROOT / "scripts" / "check_three_floor_rl_mission.py",
    )


def load_randomizer():
    return load_file_module(
        "randomize_three_floor_scene_under_test",
        ROOT / "scripts" / "randomize_three_floor_scene.py",
    )


def synthetic_three_floor_layout():
    floors = []
    room_bounds = (
        (-9.5, -1.1, 7.85, 21.88),
        (1.1, 9.5, 7.85, 21.88),
        (-9.5, -1.1, 21.88, 35.91),
        (1.1, 9.5, 21.88, 35.91),
    )
    for floor_index in range(3):
        elevation = 2.6 * floor_index
        rooms = []
        for room_index, values in enumerate(room_bounds):
            x_min, x_max, y_min, y_max = values
            room_id = "floor_{}_room_{}".format(floor_index, room_index)
            left = x_max < 0.0
            rooms.append({
                "id": room_id,
                "floor_index": floor_index,
                "bounds": {"x_min": x_min, "x_max": x_max,
                           "y_min": y_min, "y_max": y_max},
                "furniture": [{
                    "id": room_id + "_fixture",
                    "pose": [-7.0 if left else 7.0,
                             0.5 * (y_min + y_max) + 3.2,
                             elevation + 0.4, 0.0, 0.0, 0.0],
                    "size": [0.8, 0.8, 0.8],
                }],
            })
        floors.append({"floor_index": floor_index,
                       "elevation": elevation, "rooms": rooms})
    return {"floor_height": 2.6, "metadata": {}, "target_points": {},
            "floors": floors}


def load_descent_smoke_checker():
    return load_file_module(
        "check_stair_descent_smoke_under_test",
        ROOT / "scripts" / "check_stair_descent_smoke.py",
    )


def load_supervisor():
    sys.modules["rospy"] = types.ModuleType("rospy")
    gazebo_msg = types.ModuleType("gazebo_msgs.msg")
    gazebo_msg.ModelStates = type("ModelStates", (), {})
    geometry_msg = types.ModuleType("geometry_msgs.msg")
    geometry_msg.Twist = type("Twist", (), {})
    std_msg = types.ModuleType("std_msgs.msg")
    std_msg.Bool = type("Bool", (), {})
    std_msg.String = type("String", (), {})
    for package, messages in (
        ("gazebo_msgs", gazebo_msg),
        ("geometry_msgs", geometry_msg),
        ("std_msgs", std_msg),
    ):
        sys.modules[package] = types.ModuleType(package)
        sys.modules[package + ".msg"] = messages
    return load_file_module(
        "three_floor_rl_mission_supervisor_under_test",
        ROOT / "scripts" / "three_floor_rl_mission_supervisor.py",
    )


def load_mux():
    sys.modules["rospy"] = types.ModuleType("rospy")
    geometry_msg = types.ModuleType("geometry_msgs.msg")
    geometry_msg.Twist = type("Twist", (), {})
    std_msg = types.ModuleType("std_msgs.msg")
    std_msg.Bool = type("Bool", (), {})
    std_msg.Float32 = type("Float32", (), {})
    std_msg.String = type("String", (), {})
    for package, messages in (
        ("geometry_msgs", geometry_msg),
        ("std_msgs", std_msg),
    ):
        sys.modules[package] = types.ModuleType(package)
        sys.modules[package + ".msg"] = messages
    return load_file_module(
        "rl_floor_command_mux_under_test",
        ROOT / "scripts" / "rl_floor_command_mux.py",
    )


def load_sequencer():
    sys.modules["rospy"] = types.ModuleType("rospy")
    geometry_msg = types.ModuleType("geometry_msgs.msg")
    geometry_msg.PoseStamped = type("PoseStamped", (), {})
    geometry_msg.Twist = type("Twist", (), {})
    nav_msg = types.ModuleType("nav_msgs.msg")
    nav_msg.Odometry = type("Odometry", (), {})
    sensor_msg = types.ModuleType("sensor_msgs.msg")
    sensor_msg.PointCloud2 = type("PointCloud2", (), {})
    std_msg = types.ModuleType("std_msgs.msg")
    std_msg.Bool = type("Bool", (), {})
    std_msg.Float32 = type("Float32", (), {})
    std_msg.String = type("String", (), {})
    for package, messages in (
        ("geometry_msgs", geometry_msg),
        ("nav_msgs", nav_msg),
        ("sensor_msgs", sensor_msg),
        ("std_msgs", std_msg),
    ):
        sys.modules[package] = types.ModuleType(package)
        sys.modules[package + ".msg"] = messages
    return load_file_module(
        "scanplanner_three_floor_goal_sequencer_under_test",
        ROOT / "scripts" / "scanplanner_three_floor_goal_sequencer.py",
    )


def load_stair_transition():
    rospy = types.ModuleType("rospy")
    rospy.logwarn = lambda *args, **kwargs: None
    sys.modules["rospy"] = rospy
    geometry_msg = types.ModuleType("geometry_msgs.msg")
    geometry_msg.Twist = type("Twist", (), {})
    nav_msg = types.ModuleType("nav_msgs.msg")
    nav_msg.Odometry = type("Odometry", (), {})
    sensor_msg = types.ModuleType("sensor_msgs.msg")
    sensor_msg.PointCloud2 = type("PointCloud2", (), {})
    sensor_msg.Joy = type("Joy", (), {})
    sensor_msg.JointState = type("JointState", (), {})
    sensor_pc2 = types.ModuleType("sensor_msgs.point_cloud2")
    std_msg = types.ModuleType("std_msgs.msg")
    std_msg.String = type("String", (), {})
    std_msg.Bool = type("Bool", (), {})
    gazebo_msg = types.ModuleType("gazebo_msgs.msg")
    gazebo_msg.ModelStates = type("ModelStates", (), {})
    gazebo_msg.LinkStates = type("LinkStates", (), {})
    for package, messages in (
        ("geometry_msgs", geometry_msg),
        ("nav_msgs", nav_msg),
        ("sensor_msgs", sensor_msg),
        ("std_msgs", std_msg),
        ("gazebo_msgs", gazebo_msg),
    ):
        sys.modules[package] = types.ModuleType(package)
        sys.modules[package + ".msg"] = messages
    sys.modules["sensor_msgs.point_cloud2"] = sensor_pc2
    return load_file_module(
        "stair_transition_manager_under_test",
        ROOT / "scripts" / "stair_transition_manager.py",
    )


class ThreeFloorRLMissionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_config_covers_three_floors_and_twelve_rooms(self):
        checker = load_checker()
        self.assertEqual(checker.validate_config(self.config), [])
        self.assertEqual([floor["floor_number"] for floor in self.config["floors"]], [1, 2, 3])
        self.assertEqual(sum(len(floor["expected_rooms"]) for floor in self.config["floors"]), 12)
        for floor in self.config["floors"]:
            notes = {waypoint["note"] for waypoint in floor["waypoints"]}
            for room in floor["expected_rooms"]:
                self.assertIn(room + "_center", notes)
            self.assertTrue(floor["waypoints"][-1]["note"].endswith("handoff"))
        self.assertEqual(
            self.config["floors"][1]["waypoints"][0]["note"],
            "floor_1_stair_exit",
        )
        self.assertEqual(
            self.config["floors"][2]["waypoints"][0]["note"],
            "floor_2_stair_exit",
        )
        for floor in self.config["floors"][1:]:
            landing_corridor = floor["waypoints"][1]
            self.assertTrue(landing_corridor["note"].endswith(
                "_room_0_corridor"))
            self.assertTrue(landing_corridor.get("plane_forward_only"))
            self.assertTrue(landing_corridor.get("speed_scale_exempt"))

        self.assertEqual(
            self.config["return_to_lobby"]["waypoints"][0]["note"],
            "first_floor_stair_exit",
        )
        open_floor_waypoints = [
            waypoint
            for floor in self.config["floors"]
            for waypoint in floor["waypoints"]
            if float(waypoint["speed"]) == 0.70
        ]
        self.assertEqual(len(open_floor_waypoints), 24)
        long_returns = [
            waypoint
            for floor in self.config["floors"]
            for waypoint in floor["waypoints"]
            if waypoint["note"].endswith(("stair_approach", "descent_approach"))
        ]
        self.assertEqual(len(long_returns), 3)
        self.assertTrue(all(float(item["speed"]) == 0.63
                            for item in long_returns))
        self.assertTrue(all(
            min(1.65, 2.60 * float(waypoint["speed"])) == 1.65
            for waypoint in open_floor_waypoints
        ))
        handoff_waypoints = [
            waypoint
            for floor in self.config["floors"]
            for waypoint in floor["waypoints"][-2:]
        ]
        self.assertLessEqual(float(handoff_waypoints[0]["speed"]), 0.50)
        self.assertLessEqual(float(handoff_waypoints[1]["speed"]), 0.55)
        self.assertEqual(
            len(set(self.config["scene_randomization"]["floor_seeds"])), 3)
        self.assertEqual(
            self.config["red_ball_detection"]["minimum_confirmed_detections"],
            18,
        )
        self.assertTrue(
            self.config["red_ball_detection"]["require_all_scene_red_balls"])
        self.assertEqual(
            self.config["task_timing"]["maximum_exploration_duration_sec"],
            600.0,
        )
        self.assertEqual(
            self.config["floors"][2]["waypoints"][-1]["controller"],
            "direct_plane_rl",
        )
        self.assertTrue(
            self.config["floors"][2]["waypoints"][-1]["align_yaw"])
        self.assertEqual(
            self.config["floors"][2]["waypoints"][-1]["heading_tolerance"],
            0.20,
        )

    def test_flight_a_dynamic_backslide_filter_rejects_healthy_steps(self):
        module = load_stair_transition()
        manager = object.__new__(module.StairTransition)
        manager.truth_stair_heading = math.pi / 2.0
        manager.truth_flight_a_rearward_slip_speed = 0.30
        manager.truth_flight_a_rearward_slip_detect_seconds = 0.04
        manager.truth_flight_a_rearward_slip_min_height_gain = 0.40
        manager.truth_flight_a_rearward_slip_velocity_alpha = 0.70
        manager.truth_flight_a_rearward_slip_release_ratio = 0.80
        manager.truth_flight_a_rearward_speed = None
        manager.truth_flight_a_filtered_rearward_speed = None
        manager.truth_flight_a_rearward_slip_since = None
        manager.truth_flight_a_rearward_slip_latched = False
        manager.truth_flight_a_rearward_slip_events = 0
        manager.truth_flight_a_rearward_slip_last_trigger_gain = None

        # Successful stress traces peaked below 0.23 m/s after the guard's
        # 0.40 m arming gain.  Footfall-sized reversals must not latch.
        for index, rearward_speed in enumerate((0.08, 0.22, 0.18, 0.21)):
            manager.truth_twist = [0.0, -rearward_speed]
            self.assertFalse(
                manager._truth_flight_a_rearward_slip_update(
                    0.10 * index, 0.55
                )
            )
        self.assertEqual(manager.truth_flight_a_rearward_slip_events, 0)

        # The failed seed-4 trace sustained 0.31--0.35 m/s and must latch
        # only after the configured dwell, not on its first high sample.
        triggered = False
        for index, rearward_speed in enumerate((0.10, 0.35, 0.35, 0.35)):
            manager.truth_twist = [0.0, -rearward_speed]
            triggered = manager._truth_flight_a_rearward_slip_update(
                1.0 + 0.06 * index, 0.68
            )
        self.assertTrue(triggered)
        self.assertTrue(manager.truth_flight_a_rearward_slip_latched)
        self.assertEqual(manager.truth_flight_a_rearward_slip_events, 1)
        self.assertAlmostEqual(
            manager.truth_flight_a_rearward_slip_last_trigger_gain, 0.68
        )

    def test_flight_a_dynamic_backslide_ignores_fast_vertical_progress(self):
        module = load_stair_transition()
        manager = object.__new__(module.StairTransition)
        manager.truth_stair_heading = math.pi / 2.0
        manager.truth_flight_a_rearward_slip_speed = 0.30
        manager.truth_flight_a_rearward_slip_detect_seconds = 0.04
        manager.truth_flight_a_rearward_slip_min_height_gain = 0.40
        manager.truth_flight_a_rearward_slip_velocity_alpha = 0.70
        manager.truth_flight_a_rearward_slip_release_ratio = 0.80
        manager.truth_flight_a_rearward_slip_max_climb_speed = 0.12
        manager.truth_flight_a_rearward_speed = None
        manager.truth_flight_a_filtered_rearward_speed = None
        manager.truth_flight_a_filtered_climb_speed = None
        manager.truth_flight_a_rearward_slip_since = None
        manager.truth_flight_a_rearward_slip_latched = False
        manager.truth_flight_a_rearward_slip_events = 0
        manager.truth_flight_a_rearward_slip_last_trigger_gain = None

        # A strong upward footfall can project rearward horizontally without
        # being a stair backslide.  It must not start or retain the dwell.
        for index in range(4):
            manager.truth_twist = [0.0, -0.55, 0.24]
            self.assertFalse(
                manager._truth_flight_a_rearward_slip_update(
                    0.06 * index, 0.55 + 0.03 * index
                )
            )
        self.assertIsNone(manager.truth_flight_a_rearward_slip_since)
        self.assertEqual(manager.truth_flight_a_rearward_slip_events, 0)

    def test_flight_a_backslide_relanding_brakes_cross_stair_momentum(self):
        module = load_stair_transition()
        manager = object.__new__(module.StairTransition)
        manager.truth_stair_heading = math.pi / 2.0
        manager.truth_flight_a_predictive_strong_center_speed = 0.30
        manager.truth_flight_a_recovery_center_speed = 0.22
        manager.truth_flight_a_prediction_min_lateral_speed = 0.08
        manager.truth_flight_a_recovery_yaw_rate = 0.16
        manager.truth_flight_a_backslide_brake_peak_center_speed = 0.0

        # The failed final-v4 seed 3 entered the hold moving toward the stair's
        # outside edge.  Longitudinal speed must be zero, but the existing
        # centreline correction must remain active and oppose that momentum.
        manager.truth_flight_a_lateral_velocity = 0.32
        manager.truth_flight_a_command_forward_speed = 0.80
        manager.truth_flight_a_command_center_speed = -0.30
        manager.truth_flight_a_command_yaw_rate = -0.25
        vx, vy, yaw_rate, center_speed = (
            manager._truth_flight_a_backslide_relanding_control()
        )
        self.assertAlmostEqual(vx, -0.30)
        self.assertAlmostEqual(vy, 0.0)
        self.assertAlmostEqual(yaw_rate, -0.16)
        self.assertAlmostEqual(center_speed, -0.30)
        self.assertEqual(manager.truth_flight_a_command_forward_speed, 0.0)
        self.assertTrue(manager.truth_flight_a_backslide_brake_active)

        # The brake direction follows measured lateral motion, so the same
        # hold also remains safe for successful traces that slipped the other
        # way across the staircase.
        manager.truth_flight_a_lateral_velocity = -0.32
        manager.truth_flight_a_command_center_speed = -0.30
        manager.truth_flight_a_command_yaw_rate = 0.10
        vx, vy, yaw_rate, center_speed = (
            manager._truth_flight_a_backslide_relanding_control()
        )
        self.assertAlmostEqual(vx, 0.30)
        self.assertAlmostEqual(vy, 0.0)
        self.assertAlmostEqual(yaw_rate, 0.10)
        self.assertAlmostEqual(center_speed, 0.30)
        self.assertAlmostEqual(
            manager.truth_flight_a_backslide_brake_peak_center_speed, 0.30
        )

    def test_flight_b_pitch_collapse_guard_separates_failed_trace(self):
        module = load_stair_transition()
        manager = object.__new__(module.StairTransition)
        manager.truth_flight_b_pitch_rate_window = 0.18
        manager.truth_flight_b_pitch_rate_trigger = 1.60
        manager.truth_flight_b_pitch_rate_min_abs = 0.60
        manager.truth_flight_b_severe_pitch = 0.80
        manager.truth_flight_b_severe_pitch_detect_seconds = 0.10
        manager.truth_flight_b_pitch_min_height_gain = 0.08
        manager.truth_flight_b_pitch_history = []
        manager.truth_flight_b_pitch_rate = None
        manager.truth_flight_b_severe_pitch_since = None
        manager.truth_flight_b_pitch_collapse_latched = False
        manager.truth_flight_b_pitch_collapse_events = 0
        manager.truth_flight_b_pitch_last_trigger_gain = None
        manager.truth_flight_b_pitch_last_trigger_reason = None

        # A healthy learned-gait footfall may pitch deeply, but its descent
        # rate remains below the failed run's sustained collapse rate.
        healthy = (
            (0.10, -0.15, 0.09),
            (0.22, -0.32, 0.14),
            (0.34, -0.52, 0.20),
            (0.46, -0.68, 0.27),
            (0.58, -0.74, 0.34),
            (0.70, -0.55, 0.41),
        )
        for timestamp, pitch, gain in healthy:
            manager.truth_pitch = pitch
            self.assertFalse(
                manager._truth_flight_b_pitch_collapse_update(
                    timestamp, gain
                )
            )
        self.assertEqual(manager.truth_flight_b_pitch_collapse_events, 0)

        manager._reset_flight_b_pitch_collapse(clear_events=True)
        # Samples replayed from final_v2 seed-3: the trunk fell from -0.22
        # to -0.69 rad in 0.235 s despite continuing to gain height.  This
        # must trigger before the later roll/yaw Euler flip.
        failed = (
            (1.288, -0.220, 0.088),
            (1.407, -0.481, 0.163),
            (1.523, -0.689, 0.204),
        )
        triggered = False
        for timestamp, pitch, gain in failed:
            manager.truth_pitch = pitch
            triggered = manager._truth_flight_b_pitch_collapse_update(
                timestamp, gain
            )
        self.assertTrue(triggered)
        self.assertEqual(manager.truth_flight_b_pitch_collapse_events, 1)
        self.assertEqual(manager.truth_flight_b_pitch_last_trigger_reason,
                         "pitch_rate")
        self.assertAlmostEqual(
            manager.truth_flight_b_pitch_last_trigger_gain, 0.204
        )
        self.assertLessEqual(manager.truth_flight_b_pitch_rate, -1.60)

    def test_scene_randomization_is_reproducible_and_distinct_per_floor(self):
        checker = load_checker()
        randomizer = load_randomizer()
        first = randomizer.build_randomized_scene(
            synthetic_three_floor_layout(), self.config, 0)
        repeated = randomizer.build_randomized_scene(
            synthetic_three_floor_layout(), self.config, 0)
        shifted = randomizer.build_randomized_scene(
            synthetic_three_floor_layout(), self.config, 19)
        first_layout, first_config = first[0], first[1]
        self.assertEqual(first_layout, repeated[0])
        self.assertEqual(first_config, repeated[1])
        self.assertNotEqual(
            [item["pose"] for item in first_layout["danger_red_spheres"]],
            [item["pose"] for item in shifted[0]["danger_red_spheres"]],
        )
        self.assertEqual(
            first_config["scene_randomization"]["effective_floor_seeds"],
            [1103, 2207, 3301],
        )
        self.assertEqual(checker.validate_config(first_config), [])
        self.assertEqual(
            checker.validate_layout_clearance(first_config, first_layout), [])
        self.assertEqual(
            checker.validate_randomized_layout(first_config, first_layout), [])
        for floor in first_layout["floors"]:
            self.assertEqual(
                sum(len(room["red_balls"]) for room in floor["rooms"]), 6)
            for room in floor["rooms"]:
                self.assertGreaterEqual(len(room["red_balls"]), 1)
                self.assertEqual(len(room["scan_points"]), 2)
                left, right = room["scan_points"]
                self.assertGreaterEqual(
                    math.hypot(left["pose"][0] - right["pose"][0],
                               left["pose"][1] - right["pose"][1]),
                    0.80,
                )
                coverage = room["scan_coverage"]
                self.assertGreaterEqual(
                    coverage["coverage_fraction"],
                    self.config["scene_randomization"][
                        "minimum_room_scan_coverage_fraction"],
                )
                self.assertGreaterEqual(
                    coverage["robust_detection_fraction"],
                    self.config["scene_randomization"][
                        "minimum_robust_detection_fraction"],
                )
                self.assertLessEqual(
                    coverage["local_route_distance_m"], 5.40)
                self.assertLessEqual(coverage["combined_turn_rad"], 3.65)
                self.assertEqual(
                    coverage["optimization"],
                    "maximum_covered_cells_within_time_constraints",
                )
                self.assertGreater(coverage["candidate_pair_count"], 0)
                expected_pauses = sum(
                    randomizer.scan_observation_pause_count(
                        scan["scan_turn_rad"], 0.60)
                    for scan in room["scan_points"]
                )
                self.assertEqual(
                    coverage["observation_pause_count"], expected_pauses)
                self.assertAlmostEqual(
                    coverage["observation_pause_time_sec"],
                    expected_pauses * 0.60,
                )
                for scan in room["scan_points"]:
                    self.assertEqual(scan["scan_pause_interval_rad"], 0.60)
                    self.assertEqual(scan["scan_pause_dwell_sec"], 0.60)
                for ball in room["red_balls"]:
                    self.assertTrue(ball["visible_scan_ids"])
                    self.assertTrue(any(
                        randomizer.visible_in_scan(
                            ball["pose"][0], ball["pose"][1], scan) and
                        randomizer.line_of_sight_clear(
                            scan["pose"][0], scan["pose"][1],
                            ball["pose"][0], ball["pose"][1], room)
                        for scan in room["scan_points"]
                    ))
            signs = [
                1 if room["scan_points"][0]["scan_yaw_rate"] > 0 else -1
                for room in floor["rooms"]
            ]
            self.assertEqual(sum(signs), 0)
            self.assertLessEqual(sum(
                first_sign == second_sign
                for first_sign, second_sign in zip(signs, signs[1:])), 1)

    def test_invalid_route_that_skips_a_room_is_rejected(self):
        checker = load_checker()
        config = copy.deepcopy(self.config)
        config["floors"][1]["waypoints"] = [
            waypoint for waypoint in config["floors"][1]["waypoints"]
            if waypoint["note"] != "floor_1_room_2_center"
        ]
        errors = checker.validate_config(config)
        self.assertTrue(any("floor_1_room_2_center" in error for error in errors))

    def test_projected_silhouette_gap_rejects_overlap_and_accepts_separation(self):
        randomizer = load_randomizer()
        gap = float(self.config["scene_randomization"][
            "red_ball_minimum_projected_gap_rad"])
        scan = {"id": "room_scan_a", "pose": [0.0, 0.0, 0.3, 0.0]}
        radius = float(self.config["scene_randomization"]["red_ball_radius_m"])
        # Two 0.15 m balls at 3.0 m range each subtend asin(0.15/3.0) rad, so
        # bearings +/-0.35 rad keep a large projected gap while +/-0.05 rad
        # fuse the silhouettes well inside the configured 3-degree margin.
        far_apart = (
            3.0 * math.cos(0.35), 3.0 * math.sin(0.35),
            3.0 * math.cos(0.35), -3.0 * math.sin(0.35),
        )
        overlapping = (
            3.0 * math.cos(0.05), 3.0 * math.sin(0.05),
            3.0 * math.cos(0.05), -3.0 * math.sin(0.05),
        )
        self.assertTrue(randomizer.projected_silhouettes_separated(
            scan, radius, *far_apart[:2], *far_apart[2:], gap))
        self.assertFalse(randomizer.projected_silhouettes_separated(
            scan, radius, *overlapping[:2], *overlapping[2:], gap))
        # asin inputs are clamped defensively instead of raising.
        required = randomizer.required_projected_bearing_separation(
            radius, 0.05, 1.0e9, gap)
        self.assertGreater(required, math.pi / 2.0)
        self.assertEqual(
            randomizer.scan_bearing_separation_rad(
                0.0, 0.0, 1.0, 0.0, -1.0, 0.0),
            math.pi,
        )

    def test_config_validation_rejects_weakened_red_ball_constants(self):
        checker = load_checker()
        weakenings = (
            ("red_ball_wall_margin_m", 0.64),
            ("red_ball_furniture_clearance_m", 0.44),
            ("red_ball_line_of_sight_clearance_m", 0.21),
            ("red_ball_minimum_projected_gap_rad", 0.05),
            ("red_ball_scan_maximum_range_m", 5.30),
            ("red_ball_scan_maximum_range_m", 4.90),
            ("red_ball_scan_minimum_range_m", 0.50),
        )
        for key, value in weakenings:
            config = copy.deepcopy(self.config)
            config["scene_randomization"][key] = value
            self.assertTrue(
                checker.validate_config(config),
                "weakened {}={} was accepted".format(key, value),
            )
        # The exact 3-degree value in radians is accepted within tolerance.
        config = copy.deepcopy(self.config)
        config["scene_randomization"][
            "red_ball_minimum_projected_gap_rad"] = math.radians(3.0)
        self.assertEqual(checker.validate_config(config), [])

    def test_stationary_scan_pauses_are_budgeted_and_tamper_evident(self):
        checker = load_checker()
        randomizer = load_randomizer()
        self.assertEqual(
            randomizer.scan_observation_pause_count(2.573364, 0.60), 3)
        pause_locations = [
            0.60 * index for index in range(
                1,
                randomizer.scan_observation_pause_count(2.573364, 0.60) + 1)
        ]
        self.assertIn(1.20, pause_locations)

        layout, config = randomizer.build_randomized_scene(
            synthetic_three_floor_layout(), self.config, 2)[:2]
        room = layout["floors"][0]["rooms"][0]
        expected_time, expected_count, expected_pause_time = (
            checker._estimated_scan_pair_time(
                room, room["scan_points"], config["scene_randomization"]))
        coverage = room["scan_coverage"]
        self.assertAlmostEqual(
            coverage["estimated_pair_time_sec"], expected_time)
        self.assertEqual(
            coverage["observation_pause_count"], expected_count)
        self.assertAlmostEqual(
            coverage["observation_pause_time_sec"], expected_pause_time)

        for key, value in (
                ("scan_pause_interval_rad", 0.0),
                ("scan_pause_dwell_sec", 0.0)):
            weakened = copy.deepcopy(self.config)
            weakened["scene_randomization"][key] = value
            self.assertTrue(checker.validate_config(weakened))

        runtime_tampered = copy.deepcopy(config)
        runtime_scan = next(
            waypoint
            for floor in runtime_tampered["floors"]
            for waypoint in floor["waypoints"]
            if waypoint.get("scan") is True
        )
        runtime_scan["scan_pause_dwell_sec"] = 0.0
        self.assertTrue(any(
            "stationary observation schedule" in error
            for error in checker.validate_config(runtime_tampered)
        ))

        for key, value in (
                ("observation_pause_count", expected_count + 1),
                ("observation_pause_time_sec", expected_pause_time + 0.60),
                ("estimated_pair_time_sec", expected_time - expected_pause_time)):
            tampered = copy.deepcopy(layout)
            tampered["floors"][0]["rooms"][0]["scan_coverage"][key] = value
            self.assertTrue(
                checker.validate_randomized_layout(config, tampered),
                "tampered {} was accepted".format(key),
            )

    def test_randomized_layout_rejects_tampered_visibility_and_projected_gap(self):
        checker = load_checker()
        randomizer = load_randomizer()
        layout, config = randomizer.build_randomized_scene(
            synthetic_three_floor_layout(), self.config, 0)[:2]
        self.assertEqual(checker.validate_randomized_layout(config, layout), [])

        tampered = copy.deepcopy(layout)
        ball = tampered["danger_red_spheres"][0]
        ball["visible_scan_ids"] = [str(ball["room_id"]) + "_scan_z"]
        errors = checker.validate_randomized_layout(config, tampered)
        self.assertTrue(any("independently recomputed" in error
                            for error in errors))
        ball["visible_scan_ids"] = []
        errors = checker.validate_randomized_layout(config, tampered)
        self.assertTrue(any("robustly visible scan" in error
                            for error in errors))
        self.assertTrue(any("independently recomputed" in error
                            for error in errors))

        spheres = copy.deepcopy(layout)["danger_red_spheres"]
        rooms = {room["id"]: room
                 for floor in layout["floors"]
                 for room in floor["rooms"]}
        pair = None
        for first_index in range(len(spheres)):
            for second_index in range(len(spheres)):
                if first_index == second_index:
                    continue
                if spheres[first_index]["room_id"] != spheres[second_index]["room_id"]:
                    continue
                if (set(map(str, spheres[first_index]["visible_scan_ids"])) &
                        set(map(str, spheres[second_index]["visible_scan_ids"]))):
                    pair = (first_index, second_index)
                    break
            if pair:
                break
        self.assertIsNotNone(pair)
        first, second = spheres[pair[0]], spheres[pair[1]]
        shared = (set(map(str, first["visible_scan_ids"])) &
                  set(map(str, second["visible_scan_ids"]))).pop()
        room = rooms[str(first["room_id"])]
        settings = config["scene_randomization"]
        clearance = float(settings["red_ball_line_of_sight_clearance_m"])
        gap = float(settings["red_ball_minimum_projected_gap_rad"])
        radius = float(settings["red_ball_radius_m"])
        first_x, first_y = first["pose"][0], first["pose"][1]
        second_x, second_y = second["pose"][0], second["pose"][1]
        length = math.hypot(first_x - second_x, first_y - second_y)
        ux, uy = ((first_x - second_x) / length,
                  (first_y - second_y) / length)
        gap_tampered = None
        for distance in (0.30, 0.24, 0.36, 0.20, 0.42, 0.48, 0.55):
            nx = first_x - ux * distance
            ny = first_y - uy * distance
            rescans = [
                scan for scan in room["scan_points"]
                if randomizer.visible_in_scan(nx, ny, scan) and
                randomizer.line_of_sight_clear(
                    scan["pose"][0], scan["pose"][1], nx, ny, room, clearance)
            ]
            if shared not in {str(scan["id"]) for scan in rescans}:
                continue
            scan = next(item for item in room["scan_points"]
                        if str(item["id"]) == shared)
            separation = checker._bearing_separation_rad(
                scan["pose"][0], scan["pose"][1], first_x, first_y, nx, ny)
            required = checker._required_bearing_separation_rad(
                radius,
                math.hypot(first_x - scan["pose"][0], first_y - scan["pose"][1]),
                math.hypot(nx - scan["pose"][0], ny - scan["pose"][1]),
                gap)
            if separation < required:
                gap_tampered = (nx, ny, [
                    str(item["id"]) for item in rescans])
                break
        self.assertIsNotNone(gap_tampered)
        second["pose"][0] = gap_tampered[0]
        second["pose"][1] = gap_tampered[1]
        second["visible_scan_ids"] = gap_tampered[2]
        tampered = copy.deepcopy(layout)
        tampered["danger_red_spheres"] = spheres
        errors = checker.validate_randomized_layout(config, tampered)
        self.assertTrue(any("bearing space" in error for error in errors))

    def test_truth_matching_accepts_an_early_view_through_neighboring_door(self):
        checker = load_checker()
        detections = [{
            "floor": 3,
            "waypoint": "floor_2_room_0_scan_b",
            "position": [3.41, 15.50, 5.35],
        }]
        truth = [{
            "floor_index": 2,
            "room_id": "floor_2_room_1",
            "pose": [3.57, 15.54, 5.35],
        }]
        self.assertEqual(
            checker._maximum_truth_matching(detections, truth, 1.0),
            {0: 0},
        )

    def test_room_scan_clearance_rejects_furniture_collision(self):
        checker = load_checker()
        layout = {"floors": []}
        for floor in self.config["floors"]:
            rooms = []
            waypoint_by_note = {
                item["note"]: item for item in floor["waypoints"]
            }
            for room_id in floor["expected_rooms"]:
                scan = waypoint_by_note[room_id + "_center"]
                rooms.append({
                    "id": room_id,
                    "bounds": {
                        "x_min": float(scan["x"]) - 4.0,
                        "x_max": float(scan["x"]) + 4.0,
                        "y_min": float(scan["y"]) - 4.0,
                        "y_max": float(scan["y"]) + 4.0,
                    },
                    "furniture": [],
                })
            layout["floors"].append({
                "floor_index": floor["floor_index"],
                "rooms": rooms,
            })
        self.assertEqual(
            checker.validate_layout_clearance(self.config, layout), []
        )
        scan = next(
            item for item in self.config["floors"][0]["waypoints"]
            if item["note"] == "floor_0_room_0_center"
        )
        layout["floors"][0]["rooms"][0]["furniture"].append({
            "id": "blocking_table",
            "pose": [scan["x"], scan["y"], 0.4, 0.0, 0.0, 0.0],
            "size": [1.0, 1.0, 0.8],
        })
        errors = checker.validate_layout_clearance(self.config, layout)
        self.assertTrue(any("blocking_table" in error for error in errors))

    def test_plane_policy_command_turns_before_translating(self):
        supervisor = load_supervisor()
        command = supervisor.command_for_lobby_waypoint(
            (0.0, 0.0, 0.3, 0.0),
            (0.0, 2.0),
            maximum_speed=0.35,
            maximum_yaw_rate=0.45,
            position_tolerance=0.28,
            heading_release_tolerance=0.20,
        )
        self.assertEqual(command[0], 0.0)
        self.assertGreater(command[2], 0.0)
        aligned = supervisor.command_for_lobby_waypoint(
            (0.0, 0.0, 0.3, 1.57079632679),
            (0.0, 2.0),
            maximum_speed=0.35,
            maximum_yaw_rate=0.45,
            position_tolerance=0.28,
            heading_release_tolerance=0.20,
        )
        self.assertGreater(aligned[0], 0.0)
        self.assertLessEqual(aligned[0], 0.35)
        reached = supervisor.command_for_lobby_waypoint(
            (0.0, 1.9, 0.3, 1.5708),
            (0.0, 2.0),
            maximum_speed=0.35,
            maximum_yaw_rate=0.45,
            position_tolerance=0.28,
            heading_release_tolerance=0.20,
        )
        self.assertTrue(reached[-1])
        orientation_only = supervisor.command_for_lobby_waypoint(
            (-3.8, 1.73, 0.3, -1.5),
            (-3.8, 1.73),
            maximum_speed=1.10,
            maximum_yaw_rate=0.80,
            position_tolerance=0.35,
            heading_release_tolerance=0.25,
            target_yaw=0.0,
            target_heading_tolerance=0.18,
        )
        self.assertEqual(orientation_only[0], 0.0)
        self.assertGreater(orientation_only[2], 0.0)
        self.assertFalse(orientation_only[-1])
        orientation_reached = supervisor.command_for_lobby_waypoint(
            (-3.8, 1.73, 0.3, 0.1),
            (-3.8, 1.73),
            maximum_speed=1.10,
            maximum_yaw_rate=0.80,
            position_tolerance=0.35,
            heading_release_tolerance=0.25,
            target_yaw=0.0,
            target_heading_tolerance=0.18,
        )
        self.assertTrue(orientation_reached[-1])

    def test_lobby_forward_acceleration_is_bounded_after_turn(self):
        supervisor = load_supervisor()
        limiter = supervisor.limit_lobby_forward_acceleration
        self.assertAlmostEqual(limiter(0.0, 1.10, 0.05, 0.55, 0.18), 0.18)
        self.assertAlmostEqual(
            limiter(0.18, 1.10, 0.05, 0.55, 0.18), 0.2075)
        self.assertAlmostEqual(
            limiter(0.2075, 1.10, 0.05, 0.55, 0.18), 0.235)
        # The heading gate and arrival slow-down remain authoritative.
        self.assertEqual(limiter(0.80, 0.0, 0.05, 0.55, 0.18), 0.0)
        self.assertEqual(limiter(0.80, 0.30, 0.05, 0.55, 0.18), 0.30)

    def test_direct_handoff_aligns_yaw_after_reaching_position(self):
        sequencer = load_sequencer()
        turning = sequencer.direct_rl_command(
            (1.0, 2.0, 0.3, 0.0), (1.0, 2.0), 0.5,
            maximum_yaw_rate=1.15, position_tolerance=0.1,
            target_yaw=1.5708, heading_tolerance=0.12,
        )
        self.assertEqual(turning[0], 0.0)
        self.assertGreater(turning[1], 0.0)
        self.assertFalse(turning[3])
        aligned = sequencer.direct_rl_command(
            (1.0, 2.0, 0.3, 1.56), (1.0, 2.0), 0.5,
            maximum_yaw_rate=1.15, position_tolerance=0.1,
            target_yaw=1.5708, heading_tolerance=0.12,
        )
        self.assertTrue(aligned[3])

    def test_safe_bend_can_translate_but_door_gate_stays_turn_first(self):
        sequencer = load_sequencer()
        pose = (0.0, 0.0, 0.3, 0.0)
        target = (2.0, 1.5)  # heading error ~= 0.644 rad
        strict = sequencer.direct_rl_command(
            pose, target, 1.2, maximum_yaw_rate=1.15,
            translation_heading_limit=0.16)
        self.assertEqual(strict[0], 0.0)
        self.assertGreater(strict[1], 0.0)
        rounded = sequencer.direct_rl_command(
            pose, target, 1.2, maximum_yaw_rate=1.15,
            translation_heading_limit=0.72)
        self.assertGreater(rounded[0], 0.0)
        self.assertLessEqual(rounded[0], 1.2)
        self.assertGreater(rounded[1], 0.0)

    def test_holonomic_bend_velocity_stays_on_audited_world_segment(self):
        sequencer = load_sequencer()
        pose = (0.0, 0.0, 0.3, 0.15)
        target = (1.0, 2.0)
        forward, lateral, yaw_rate, distance, reached = (
            sequencer.direct_holonomic_rl_command(
                pose, target, 1.65, maximum_lateral_speed=0.75))
        self.assertFalse(reached)
        self.assertGreater(forward, 0.0)
        self.assertGreater(lateral, 0.0)
        self.assertGreater(yaw_rate, 0.0)
        # Convert the body command back to world coordinates.  Its bearing
        # must equal the audited current-to-target line despite component caps.
        world_x = math.cos(pose[3]) * forward - math.sin(pose[3]) * lateral
        world_y = math.sin(pose[3]) * forward + math.cos(pose[3]) * lateral
        self.assertAlmostEqual(
            math.atan2(world_y, world_x), math.atan2(2.0, 1.0), places=7)
        self.assertAlmostEqual(distance, math.sqrt(5.0))

    def test_entrance_command_keeps_doorway_heading_and_corrects_side_drift(self):
        sequencer = load_sequencer()
        command = sequencer.direct_entrance_rl_command(
            (0.0, -3.29, 0.31, 1.5708), (0.0, -2.05), 1.5708,
            maximum_speed=0.348)
        self.assertGreater(command[0], 0.0)
        self.assertAlmostEqual(command[1], 0.0, places=4)
        self.assertAlmostEqual(command[2], 0.0, places=4)

        # Replay the terminal pose from the seed2 entrance failure.  The old
        # point-at-target controller requested positive yaw toward 2.66 rad;
        # the fixed-axis controller must first turn back toward 1.57 rad.
        failed_pose = (0.5168, -2.3224, 0.3529, 2.4088)
        recovering = sequencer.direct_entrance_rl_command(
            failed_pose, (0.0, -2.05), 1.5708,
            maximum_speed=0.348)
        self.assertEqual(recovering[0], 0.0)
        self.assertEqual(recovering[1], 0.0)
        self.assertLess(recovering[2], 0.0)
        self.assertFalse(recovering[-1])

        aligned_recovery = sequencer.direct_entrance_rl_command(
            (0.5168, -2.3224, 0.3529, 1.5708), (0.0, -2.05), 1.5708,
            maximum_speed=0.348)
        self.assertGreater(aligned_recovery[0], 0.0)
        self.assertGreater(aligned_recovery[1], 0.0)
        self.assertLessEqual(aligned_recovery[1], 0.12)
        self.assertFalse(aligned_recovery[-1])

        reached = sequencer.direct_entrance_rl_command(
            (0.01, -2.52, 0.32, 1.59), (0.0, -2.05), 1.5708,
            maximum_speed=0.348)
        self.assertTrue(reached[-1])

    def test_attitude_guard_debounces_a_single_transient_sample(self):
        sequencer = load_sequencer()
        since, failed = sequencer.attitude_loss_debounce(
            None, 0.546, 0.55, 10.0, 0.60)
        self.assertFalse(failed)
        since, failed = sequencer.attitude_loss_debounce(
            since, 0.545, 0.55, 10.4, 0.60)
        self.assertFalse(failed)
        since, failed = sequencer.attitude_loss_debounce(
            since, 0.540, 0.55, 10.7, 0.60)
        self.assertTrue(failed)
        since, failed = sequencer.attitude_loss_debounce(
            since, 0.90, 0.55, 10.8, 0.60)
        self.assertIsNone(since)
        self.assertFalse(failed)

    def test_plane_attitude_guard_preemptively_unloads_motion(self):
        sequencer = load_sequencer()
        scale = sequencer.plane_attitude_speed_scale
        self.assertEqual(scale(None), 0.0)
        self.assertEqual(scale(0.99), 1.0)
        self.assertAlmostEqual(scale(0.935), 0.5, places=6)
        self.assertEqual(scale(0.90), 0.0)
        self.assertEqual(scale(-0.647), 0.0)

    def test_required_policy_hot_switch_sequence(self):
        supervisor = load_supervisor()
        valid = [
            {"policy": "stair"}, {"policy": "plane"},
            {"policy": "stair"}, {"policy": "plane"},
            {"policy": "stair"}, {"policy": "plane"},
            {"policy": "stair"}, {"policy": "plane"},
        ]
        self.assertTrue(supervisor.policy_sequence_satisfied(valid))
        self.assertFalse(supervisor.policy_sequence_satisfied(valid[:-1]))

    def test_entrance_step_is_a_distinct_timed_stage(self):
        supervisor = load_supervisor()
        self.assertEqual(
            supervisor.task_stage_for_route_state({
                "phase": "ENTER_MAIN_ENTRANCE_STAIR_RL", "floor": 1,
            }),
            "entrance_step_stair_rl",
        )
        self.assertEqual(
            supervisor.task_stage_for_route_state({
                "phase": "EXPLORE_FLOOR", "floor": 1,
            }),
            "floor_1_exploration",
        )

    def test_floor_mux_limits_plane_rl_and_stops_translation_while_turning(self):
        mux = load_mux()
        shaped = mux.shape_command(
            1.4, 0.8, 0.50, speed_limit=0.42,
            max_forward=0.65, max_lateral=0.18,
            yaw_translation_stop=0.42,
        )
        self.assertEqual(shaped[0], 0.0)
        self.assertEqual(shaped[1], 0.0)
        self.assertLessEqual(abs(shaped[2]), 0.55)
        translating = mux.shape_command(
            1.4, 0.8, 0.0, speed_limit=0.42,
            max_forward=0.65, max_lateral=0.18,
        )
        self.assertLessEqual(
            (translating[0] ** 2 + translating[1] ** 2) ** 0.5, 0.420001)

    @staticmethod
    def _write(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_acceptance_requires_all_rooms_stairs_policy_sequence_and_lobby(self):
        checker = load_checker()
        randomizer = load_randomizer()
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory)
            (runtime_layout, runtime_config, _furniture, _danger, red_truth,
             _scans, _seeds) = randomizer.build_randomized_scene(
                synthetic_three_floor_layout(), self.config, 0)
            self._write(result / "layout_metadata.json", runtime_layout)
            tour_paths = (
                result / "tour_summary.json",
                result / "second_floor" / "tour_summary.json",
                result / "third_floor" / "tour_summary.json",
            )
            for floor, path in zip(runtime_config["floors"], tour_paths):
                entrance_records = []
                if floor["floor_number"] == 1:
                    stair_policy = runtime_config["runtime"]["stair_policy"]
                    entrance_records = [
                        {"note": note, "success": True,
                         "scan_completed": False,
                         "controller": "direct_forward_stair_rl",
                         "policy": stair_policy}
                        for note in (
                            "main_entrance_step_below",
                            "main_entrance_apron",
                            "main_entrance_threshold_clear",
                        )
                    ]
                self._write(path, {
                    "planner": "SCAN-Planner",
                    "motion_backend": "plane_rl",
                    "entrance_motion_backend": (
                        "stair_rl" if floor["floor_number"] == 1 else None),
                    "entrance_policy": (
                        runtime_config["runtime"]["stair_policy"]
                        if floor["floor_number"] == 1 else None),
                    "success": True,
                    "termination_reason": "WAYPOINT_TOUR_COMPLETE",
                    "waypoints": entrance_records + [
                        {"note": waypoint["note"], "success": True,
                         "scan_completed": True,
                         "scan_rotation_rad": waypoint["scan_turn_rad"]}
                        for waypoint in floor["waypoints"]
                        if waypoint.get("scan") is True
                    ],
                })
            self._write(result / "logs" / "stair_transition.json",
                        {"phase": "SECOND_FLOOR_HANDOFF_COMPLETE"})
            self._write(result / "logs" / "second_to_third_floor_stair_transition.json",
                        {"phase": "THIRD_FLOOR_HANDOFF_COMPLETE"})
            self._write(result / "logs" / "stair_descent.json",
                        {"phase": "FIRST_FLOOR_RETURNED"})
            self._write(result / "first_floor_returned.json",
                        {"phase": "FIRST_FLOOR_RETURNED"})
            required_stages = runtime_config["task_timing"]["required_stages"]
            self._write(result / "mission_stage_timing.json", {
                "schema": "scanplanner_mission_stage_timing_v1",
                "status": "completed",
                "preparation_excluded": True,
                "maximum_duration_sec": 600.0,
                "budget_met": True,
                "total_duration_sec": 512.0,
                "spawn_truth_pose_at_start": [0.02, -3.18, 0.32, 1.57],
                "stages": [
                    {"name": name, "duration_sec": 50.0,
                     "sim_duration_sec": 48.0}
                    for name in required_stages
                ],
            })
            self._write(result / "red_ball_detections.json", {
                "schema": "scanplanner_red_ball_detections_v1",
                "status": "completed",
                "preparation_excluded": True,
                "frames_processed": 120,
                "confirmed_count": len(red_truth),
                "detections": [
                    {
                        "track_id": index + 1,
                        "position": list(item["pose"][:3]),
                        "first_confirmed_elapsed_sec": 42.5 + index,
                        "stage": "floor_{}_exploration".format(
                            int(item["floor_index"]) + 1),
                        "floor": int(item["floor_index"]) + 1,
                        "evidence_frames": 3,
                        "waypoint": item["room_id"] + "_scan_a",
                        "scan_active": True,
                    }
                    for index, item in enumerate(red_truth)
                ],
            })
            self._write(result / "three_floor_rl_mission_summary.json", {
                "schema": "scanplanner_three_floor_rl_summary_v1",
                "status": "completed",
                "policy_events": [
                    {"policy": "stair"}, {"policy": "plane"},
                    {"policy": "stair"}, {"policy": "plane"},
                    {"policy": "stair"}, {"policy": "plane"},
                    {"policy": "stair"}, {"policy": "plane"},
                ],
                "final_truth_pose": [0.0, 4.0, 0.31, 0.0],
                "return_waypoints": [{"index": 0}, {"index": 1}],
            })
            acceptance = checker.check_results(runtime_config, str(result))
            self.assertTrue(acceptance["passed"], acceptance["failure_reasons"])
            self.assertTrue(all(acceptance["checks"].values()))

            first_tour = json.loads(
                tour_paths[0].read_text(encoding="utf-8")
            )
            first_scan = next(
                item for item in first_tour["waypoints"]
                if item.get("scan_completed") is True
            )
            first_scan["scan_rotation_rad"] = 0.1
            self._write(tour_paths[0], first_tour)
            acceptance = checker.check_results(runtime_config, str(result))
            self.assertFalse(acceptance["passed"])
            self.assertFalse(acceptance["checks"]["floor_1_tour"])
            first_scan["scan_rotation_rad"] = 1.90
            self._write(tour_paths[0], first_tour)

            first_tour["waypoints"][0]["controller"] = "direct_forward_plane_rl"
            self._write(tour_paths[0], first_tour)
            acceptance = checker.check_results(runtime_config, str(result))
            self.assertFalse(acceptance["passed"])
            self.assertFalse(acceptance["checks"]["main_entrance_step_stair_rl"])
            first_tour["waypoints"][0]["controller"] = "direct_forward_stair_rl"
            self._write(tour_paths[0], first_tour)

            red_path = result / "red_ball_detections.json"
            red_payload = json.loads(red_path.read_text(encoding="utf-8"))
            missing_detection = red_payload["detections"].pop()
            red_payload["confirmed_count"] = len(red_payload["detections"])
            self._write(red_path, red_payload)
            acceptance = checker.check_results(runtime_config, str(result))
            self.assertFalse(acceptance["passed"])
            self.assertFalse(
                acceptance["checks"]["all_scene_red_balls_detected"])
            red_payload["detections"].append(missing_detection)
            red_payload["confirmed_count"] = len(red_payload["detections"])
            self._write(red_path, red_payload)

            summary_path = result / "three_floor_rl_mission_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["final_truth_pose"] = [0.0, 20.0, 0.31, 0.0]
            self._write(summary_path, summary)
            acceptance = checker.check_results(runtime_config, str(result))
            self.assertFalse(acceptance["passed"])
            self.assertFalse(acceptance["checks"]["final_pose_in_first_floor_lobby"])

    def test_descent_smoke_acceptance_requires_physical_two_floor_truth_drop(self):
        checker = load_descent_smoke_checker()
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory)
            trace = []
            phases = (
                (0, "STAIR_DESCENT_FLIGHT_B", 5.65),
                (0, "STAIR_DESCENT_TURN", 4.45),
                (0, "STAIR_DESCENT_FLIGHT_A", 3.15),
                (1, "STAIR_DESCENT_SEGMENT_TURN", 3.10),
                (1, "STAIR_DESCENT_FLIGHT_B", 3.00),
                (1, "STAIR_DESCENT_TURN", 1.85),
                (1, "STAIR_DESCENT_FLIGHT_A", 0.72),
                (1, "FIRST_FLOOR_RETURNED", 0.68),
            )
            for segment, phase, z_value in phases:
                trace.append({
                    "segment": segment,
                    "phase": phase,
                    "pose_source": "gazebo_truth",
                    "truth_x": -3.25,
                    "truth_y": 1.55,
                    "truth_z": z_value,
                    "roll": 0.02,
                    "pitch": 0.04,
                })
            self._write(result / "logs" / "stair_descent.json", {
                "phase": "FIRST_FLOOR_RETURNED",
                "policy": "/tmp/policy_act_inference_stair.pt",
                "trace": trace,
            })
            self._write(
                result / "logs" / "third_to_second_floor_stair_transition.json",
                {"phase": "SECOND_FLOOR_DESCENT_REACHED"},
            )
            self._write(
                result / "logs" / "second_to_first_floor_stair_transition.json",
                {"phase": "FIRST_FLOOR_RETURNED"},
            )
            self._write(result / "first_floor_returned.json", {
                "phase": "FIRST_FLOOR_RETURNED",
                "descent_total_drop": 2.42,
            })
            acceptance = checker.check_results(str(result))
            self.assertTrue(acceptance["passed"], acceptance["failure_reasons"])

            primary_path = result / "logs" / "stair_descent.json"
            primary = json.loads(primary_path.read_text(encoding="utf-8"))
            primary["trace"][-1]["truth_z"] = 2.2
            self._write(primary_path, primary)
            acceptance = checker.check_results(str(result))
            self.assertFalse(acceptance["passed"])
            self.assertFalse(acceptance["checks"]["physical_two_floor_drop"])

            primary["trace"][-1]["truth_z"] = 0.31
            primary["trace"][-1]["truth_x"] = 0.0
            self._write(primary_path, primary)
            acceptance = checker.check_results(str(result))
            self.assertFalse(acceptance["passed"])
            self.assertFalse(
                acceptance["checks"]["final_pose_on_first_floor_landing"]
            )

    def test_room_legs_get_the_same_bounded_stall_escape_as_entrance_legs(self):
        launch = (ROOT / "launch" / "scanplanner_three_floor_rl.launch").read_text(
            encoding="utf-8")
        self.assertIn(
            '<param name="stall_recovery_max_attempts" value="2"/>', launch)
        sequencer_source = (
            ROOT / "scripts" / "scanplanner_three_floor_goal_sequencer.py"
        ).read_text(encoding="utf-8")
        # The escape must not be reachable only from an entrance waypoint:
        # seed 205 lost floor_1_room_3_g4 to a deadlock on open floor.
        self.assertIn("self._stall_recovery_max_attempts", sequencer_source)
        self.assertNotIn(
            "if (entrance_mode and stall_recoveries <", sequencer_source)

    def test_launch_and_runners_use_real_rl_and_isolated_container(self):
        launch = (ROOT / "launch" / "scanplanner_three_floor_rl.launch").read_text(
            encoding="utf-8"
        )
        descent_controller = (
            ROOT / "launch" / "stair_descent_controller.launch"
        ).read_text(encoding="utf-8")
        descent_smoke = (
            ROOT / "launch" / "stair_descent_physical_smoke.launch"
        ).read_text(encoding="utf-8")
        controller_launch = (
            ROOT / "launch" / "unitree_rl_controller.launch"
        ).read_text(encoding="utf-8")
        in_container = (ROOT / "scripts" / "run_scanplanner_three_floor_rl.sh").read_text(
            encoding="utf-8"
        )
        host = (ROOT.parents[4] / "run_three_floor_rl_docker.sh").read_text(
            encoding="utf-8"
        )
        smoke_host = (
            ROOT.parents[4] / "run_stair_descent_smoke_docker.sh"
        ).read_text(encoding="utf-8")
        supervisor = (ROOT / "scripts" / "three_floor_rl_mission_supervisor.py").read_text(
            encoding="utf-8"
        )
        sequencer = (
            ROOT / "scripts" / "scanplanner_three_floor_goal_sequencer.py"
        ).read_text(encoding="utf-8")
        scan_bridge = (ROOT / "scripts" / "scan_to_map.py").read_text(
            encoding="utf-8"
        )
        stair_manager = (
            ROOT / "scripts" / "stair_transition_manager.py"
        ).read_text(encoding="utf-8")

        self.assertIn("policy_act_inference_plane.pt", launch)
        self.assertIn('$(find scan_planner)/launch/simenv_scan.launch', launch)
        self.assertIn('type="scanplanner_three_floor_goal_sequencer.py"', launch)
        self.assertIn('type="scanplanner_pointcloud_bridge.py"', launch)
        self.assertNotIn('type="pointcloud2livox.py"', launch)
        self.assertNotIn('type="scan_to_map.py"', launch)
        self.assertNotIn('type="scanplanner_record.py"', launch)
        self.assertIn('name="output" value="/registered_scan"', launch)
        self.assertIn('name="slam_startup_delay_seconds" value="1200.0"', launch)
        self.assertIn('type="rl_floor_command_mux.py"', launch)
        self.assertIn('name="command_topic" value="/scanplanner/planner_cmd_vel"', launch)
        self.assertIn('name="manual_goal_use_current_height" value="true"', launch)
        self.assertIn('name="max_vel" value="1.65"', launch)
        self.assertIn('name="max_forward" value="1.65"', launch)
        self.assertIn('name="speed_scale" value="2.60"', launch)
        self.assertIn('name="maximum_floor_speed" value="1.65"', launch)
        self.assertIn('name="room_scan_timeout_sec" value="30.0"', launch)
        self.assertIn('name="room_scan_yaw_rate" value="2.80"', launch)
        self.assertIn('name="room_scan_turn_rad" value="1.90"', launch)
        self.assertIn('name="robot_y" value="-3.2"', launch)
        self.assertIn('type="danger_detector.py"', launch)
        self.assertNotIn('name="gazebo_recording_camera_bridge"', launch)
        self.assertIn('/recording_camera/image_raw', launch)
        self.assertIn('name="camera_ready_topic" value="/simenv/recording_camera_ready"', launch)
        self.assertIn(
            'name="scan_active_topic" value="/scanplanner/room_scan_active"',
            launch,
        )
        self.assertIn('name="maximum_processing_rate_hz" value="0.0"', launch)
        self.assertIn('name="confirm_count" value="3"', launch)
        self.assertIn('name="front_camera_update_rate" value="10.0"', launch)
        self.assertIn('name="enable_realsense" value="false"', launch)
        self.assertIn('name="enable_front_camera" value="true"', launch)
        self.assertIn('name="use_depth" value="false"', launch)
        self.assertIn("room_scan_final_observation_dwell", sequencer)
        final_hold = sequencer.index("self._publish_scan(True, 0.0)")
        disable_scan = sequencer.index(
            "self._publish_room_scan_active(False)", final_hold)
        self.assertLess(final_hold, disable_scan)
        self.assertIn('name="open_mapping_entrance" value="true"', launch)
        self.assertIn('name="handoff_timeout_sec" value="300.0"', launch)
        self.assertIn('name="two_flight_stair" value="true"', launch)
        self.assertIn(
            'name="truth_flight_a_policy_use_default_stand_target" value="false"',
            launch,
        )
        self.assertIn(
            'name="truth_flight_a_policy_require_fixed_stand_ready" value="true"',
            launch,
        )
        self.assertIn(
            'name="truth_flight_a_policy_fast_fixed_stand" value="true"',
            launch,
        )
        self.assertIn(
            'name="policy_switch_fast_fixed_stand" value="true"',
            descent_controller,
        )
        self.assertEqual(
            2, launch.count('name="landing_turn_speed_rps" value="1.20"')
        )
        self.assertEqual(
            2,
            launch.count(
                'name="truth_landing_recenter_speed_mps" value="0.75"'
            ),
        )
        self.assertEqual(
            2,
            launch.count(
                'name="truth_flight_a_policy_fast_fixed_stand_duration_seconds" value="0.50"'
            ),
        )
        self.assertEqual(
            2,
            launch.count(
                'name="truth_flight_a_policy_settle_seconds" value="0.35"'
            ),
        )
        self.assertEqual(
            2,
            launch.count(
                'name="pre_ascent_rl_fast_blend_seconds" value="0.50"'
            ),
        )
        self.assertEqual(
            2,
            launch.count(
                'name="pre_ascent_rl_fast_zero_hold_seconds" value="0.15"'
            ),
        )
        self.assertEqual(
            2,
            launch.count('name="second_floor_stable_seconds" value="0.50"'),
        )
        self.assertEqual(
            2,
            launch.count(
                'name="plane_fast_takeover_blend_seconds" value="0.50"'
            ),
        )
        self.assertEqual(
            2,
            launch.count(
                'name="plane_fast_takeover_zero_hold_seconds" value="0.15"'
            ),
        )
        self.assertEqual(
            1,
            launch.count('name="lobby_heading_stable_sec" value="0.30"'),
        )
        self.assertEqual(
            1,
            launch.count(
                'name="lobby_linear_acceleration_mps2" value="0.55"'
            ),
        )
        self.assertEqual(
            1,
            launch.count(
                'name="lobby_initial_forward_speed_mps" value="0.18"'
            ),
        )
        self.assertIn(
            'name="manager_release_settle_sec" value="0.10"', launch
        )
        self.assertIn(
            'name="entrance_fixed_heading_release_rad" value="0.20"', launch
        )
        self.assertIn(
            'name="entrance_max_lateral_speed_mps" value="0.12"', launch
        )
        self.assertIn(
            'name="entrance_progress_timeout_sec" value="5.0"', launch
        )
        self.assertIn(
            'name="entrance_recovery_max_attempts" value="2"', launch
        )
        self.assertEqual(
            2,
            launch.count(
                'name="truth_flight_a_policy_capture_max_joint_velocity_rms" value="0.18"'
            ),
        )
        self.assertEqual(
            2,
            launch.count(
                'name="truth_flight_a_recovery_forward_speed_mps" value="0.60"'
            ),
        )
        self.assertEqual(
            2,
            launch.count(
                'name="truth_flight_a_strong_recovery_forward_speed_mps" value="0.40"'
            ),
        )
        self.assertEqual(
            2,
            launch.count(
                'name="truth_flight_b_recovery_forward_speed_mps" value="0.60"'
            ),
        )
        self.assertEqual(
            2,
            launch.count(
                'name="truth_flight_b_strong_recovery_forward_speed_mps" value="0.40"'
            ),
        )
        for parameter, value in (
            ("truth_flight_a_rearward_slip_speed_mps", "0.30"),
            ("truth_flight_a_backslide_recovery_hold_seconds", "1.50"),
            ("truth_flight_a_backslide_recovery_ramp_seconds", "1.00"),
            ("truth_flight_a_backslide_recovery_ramp_speed_mps", "0.40"),
            ("truth_flight_a_backslide_recovery_max_attempts", "2"),
            ("truth_flight_b_pitch_rate_window_sec", "0.18"),
            ("truth_flight_b_pitch_rate_trigger_rps", "1.60"),
            ("truth_flight_b_pitch_rate_min_abs_rad", "0.60"),
            ("truth_flight_b_severe_pitch_rad", "0.80"),
            ("truth_flight_b_severe_pitch_detect_seconds", "0.10"),
            ("truth_flight_b_pitch_min_height_gain_m", "0.08"),
            ("truth_flight_b_pitch_recovery_max_attempts", "2"),
        ):
            self.assertEqual(
                2,
                launch.count(
                    'name="{}" value="{}"'.format(parameter, value)
                ),
            )
        self.assertIn("def _truth_flight_a_rearward_slip_update", stair_manager)
        self.assertIn(
            "STAIR_FLIGHT_A_BACKSLIDE_EXHAUSTED", stair_manager
        )
        self.assertIn("flight_a_filtered_rearward_speed", stair_manager)
        self.assertIn("flight_a_backslide_recovery_active", stair_manager)
        self.assertIn('pkg="simenv_bridge" type="stair_transition_manager.py"', launch)
        self.assertTrue((ROOT / "scripts" / "stair_transition_manager.py").is_file())
        self.assertIn(
            '$(find simenv_bridge)/launch/stair_descent_controller.launch', launch
        )
        self.assertIn(
            'pkg="simenv_bridge" type="stair_descent_manager.py"',
            descent_controller,
        )
        self.assertTrue((ROOT / "scripts" / "stair_descent_manager.py").is_file())
        for launch_text in (
            launch, descent_controller, descent_smoke, controller_launch
        ):
            for node in ET.fromstring(launch_text).iter("node"):
                parameters = [item.attrib["name"] for item in node.findall("param")]
                self.assertEqual(
                    len(parameters), len(set(parameters)),
                    "duplicate private parameter in node {}".format(node.attrib.get("name")),
                )
        for parameter, value in (
            ("truth_flight_a_require_fresh_ready", "true"),
            ("truth_flight_a_climb_ramp_seconds", "1.20"),
            ("truth_flight_b_rearward_slip_speed_mps", "0.30"),
            ("truth_flight_b_max_yaw_rate_rps", "0.10"),
        ):
            self.assertIn(
                'name="{}" value="{}"'.format(parameter, value),
                launch + descent_controller,
            )
        self.assertNotIn("waypoint_tour_mode", launch)
        node_types = {
            node.attrib.get("type")
            for node in ET.fromstring(launch).iter("node")
        }
        self.assertNotIn("body_cmd_vel_driver.py", node_types)
        self.assertNotIn("set_model_state", supervisor)
        self.assertIn('"ATTITUDE_LOST"', supervisor)
        self.assertIn('"ATTITUDE_LOST"', sequencer)
        self.assertIn("/simenv/rl_policy_request", supervisor)
        self.assertIn("if not input_is_world and src_frame != target", scan_bridge)
        self.assertIn("msg.header.frame_id = target", scan_bridge)
        self.assertIn("three_floor_rl_acceptance.json", in_container)
        self.assertIn('RED_RECORD="$RESULTS_DIR/red_ball_detections.json"', in_container)
        self.assertIn('[ "$RED_STATUS" = "completed" ] && break', in_container)
        self.assertLess(
            in_container.index('RED_RECORD="$RESULTS_DIR/red_ball_detections.json"'),
            in_container.index("stop_launch\nstop_xvfb"),
        )
        self.assertIn("rospack find scan_planner", in_container)
        self.assertIn("--only-pkg-with-deps scan_planner", in_container)
        self.assertIn('find_node("simenv_bridge", "stair_transition_manager.py")', in_container)
        self.assertIn('find_node("simenv_bridge", "stair_descent_manager.py")', in_container)
        self.assertIn("load_config_default", in_container)
        self.assertIn("missing ROS node executables", in_container)
        self.assertIn('PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"', in_container)
        self.assertIn('DESCENT_SMOKE_ONLY="${DESCENT_SMOKE_ONLY:-0}"', in_container)
        self.assertIn("stair_descent_physical_smoke.launch", in_container)
        self.assertIn("catkin build unitree_guide --no-deps", in_container)
        self.assertIn("lacks hybrid policy-device support", in_container)
        self.assertIn("prepare_three_floor_scene.py", in_container)
        self.assertIn("randomize_three_floor_scene.py", in_container)
        self.assertIn("THREE_FLOOR_SEED_OFFSET", in_container)
        self.assertIn("THREE_FLOOR_WORLD_FILE", in_container)
        self.assertIn("rostest simenv_bridge mock_three_floor_ros_integration.test", in_container)
        self.assertIn("docker run -d", host)
        self.assertIn('PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"', host)
        self.assertIn('DESCENT_SMOKE_ONLY="${DESCENT_SMOKE_ONLY:-0}"', host)
        self.assertIn('ALLOW_CONCURRENT_SIM="${ALLOW_CONCURRENT_SIM:-0}"', host)
        self.assertIn('MIN_FREE_GPU_MIB="${MIN_FREE_GPU_MIB:-1800}"', host)
        self.assertIn('THREE_FLOOR_SEED_OFFSET="${THREE_FLOOR_SEED_OFFSET:-0}"', host)
        self.assertIn('GPU_ARGS=(--gpus all)', host)
        self.assertIn("readonly", host)
        self.assertIn("dst=/workspace/SCAN-Planner/src,readonly", host)
        self.assertNotIn("--network host", host)
        self.assertNotIn("simenv-new-repro", host)
        self.assertIn("DESCENT_SMOKE_ONLY=1", smoke_host)
        self.assertIn("stair_descent_controller.launch", descent_smoke)
        self.assertIn("unitree_rl_controller.launch", launch)
        self.assertIn("unitree_rl_controller.launch", descent_smoke)
        self.assertIn('name="start_controller" value="false"', launch)
        self.assertIn('name="start_controller" value="false"', descent_smoke)
        self.assertIn('type="classic_trotting_controller.py"', controller_launch)
        self.assertIn('name="UNITREE_RL_DEVICE" value="$(arg rl_device)"', controller_launch)
        self.assertIn("UNITREE_RL_PRELOAD_PLANE_POLICY", controller_launch)
        self.assertIn("UNITREE_RL_PRELOAD_STAIR_POLICY", controller_launch)
        self.assertIn(
            'name="preload_stair_policy" value="$(arg stair_policy)"',
            launch,
        )
        self.assertIn('name="OMP_NUM_THREADS" value="1"', controller_launch)
        self.assertIn('name="MKL_NUM_THREADS" value="1"', controller_launch)
        self.assertIn('name="skip_entry_guide" value="true"', descent_smoke)
        self.assertIn('name="robot_x" value="-2.665"', descent_smoke)
        self.assertIn('name="robot_y" value="1.625"', descent_smoke)
        self.assertIn('name="robot_z" value="5.71"', descent_smoke)
        self.assertIn(
            'name="descent_speed_mps" value="$(arg descent_speed_mps)"',
            descent_controller,
        )
        self.assertIn(
            'name="truth_descent_step_pause_seconds" value="0.00"',
            descent_controller,
        )
        self.assertIn(
            'name="policy_switch_fast_fixed_stand_duration_seconds" value="0.50"',
            descent_controller,
        )
        self.assertIn(
            'name="policy_switch_fast_fixed_stand_minimum_gain_ratio" value="0.75"',
            descent_controller,
        )
        self.assertEqual(
            launch.count(
                'name="truth_flight_a_policy_fast_fixed_stand_minimum_gain_ratio" '
                'value="0.75"'
            ),
            2,
        )
        for parameter, value in (
                ("pre_descent_stand_seconds", "0.25"),
                ("policy_switch_settle_seconds", "0.45"),
                ("policy_warmup_seconds", "0.75"),
                ("truth_descent_landing_settle_seconds", "1.50"),
                ("truth_descent_landing_turn_speed_rps", "1.20"),
                ("truth_descent_final_landing_settle_seconds", "1.0")):
            self.assertIn(
                'name="{}" value="{}"'.format(parameter, value),
                descent_controller,
            )
        self.assertIn(
            'name="truth_descent_flight_center_correction_speed_mps" value="0.50"',
            descent_controller,
        )
        self.assertIn(
            'name="truth_descent_flight_initial_speed_mps" value="0.55"',
            descent_controller,
        )
        self.assertIn(
            'name="truth_descent_flight_tilt_slowdown_rad" value="0.48"',
            descent_controller,
        )
        self.assertIn(
            'name="truth_descent_flight_tilt_speed_mps" value="0.55"',
            descent_controller,
        )
        self.assertIn(
            'name="truth_descent_flight_tilt_hold_sec" value="0.60"',
            descent_controller,
        )
        self.assertIn(
            'name="truth_descent_flight_heading_correction_yaw_rate_rps" value="0.30"',
            descent_controller,
        )
        self.assertIn(
            'name="truth_descent_landing_position_tolerance_m" value="0.12"',
            descent_controller,
        )
        self.assertIn(
            'name="truth_descent_segment_turn_position_tolerance_m" value="0.16"',
            descent_controller,
        )
        self.assertIn(
            'name="truth_descent_segment_turn_alignment_progress_timeout_sec" value="3.0"',
            descent_controller,
        )
        self.assertIn('name="descent_speed_mps" default="0.80"', descent_smoke)
        self.assertIn('-e "DESCENT_SPEED_MPS=$DESCENT_SPEED_MPS"', host)
        self.assertNotIn("set_model_state", descent_smoke)

        controller_source = (
            ROOT / "vendor" / "unitree_guide_controller" / "src" / "FSM" /
            "State_RL_test.cpp"
        ).read_text(encoding="utf-8")
        self.assertIn('requested_device == "hybrid"', controller_source)
        self.assertIn("policy_reloaded:", controller_source)
        self.assertIn("policy_cache_hit:", controller_source)
        self.assertIn("preloadConfiguredPolicies", controller_source)
        self.assertNotIn("torch::jit::load(requested)", controller_source)

        descent_manager = (ROOT / "scripts" / "stair_descent_manager.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("landing_heading_captured", descent_manager)
        self.assertIn(
            "max(0.0, self.landing_minimum_x_speed)", descent_manager
        )
        self.assertIn("'command_axis_speed_mps'", descent_manager)
        self.assertIn("'severe_tilt_speed_cap_active'", descent_manager)

        fixed_stand_source = (
            ROOT / "vendor" / "unitree_guide_controller" / "src" / "FSM" /
            "State_FixedStand.cpp"
        ).read_text(encoding="utf-8")
        self.assertIn("fixed_stand_fast_ready_enabled", fixed_stand_source)
        self.assertIn("fixed_stand_fast_minimum_gain_ratio", fixed_stand_source)
        self.assertIn("_minimumGainRatio", fixed_stand_source)
        self.assertIn(
            "truth_flight_a_policy_fast_fixed_stand_minimum_gain_ratio",
            stair_manager,
        )

        fsm = (ROOT.parents[4] / "workspace" / "simenv_reproduce" /
               "SCAN-Planner-main" / "src" / "planner" /
               "plan_manage" / "src" / "scan_replan_fsm.cpp").read_text(
                   encoding="utf-8")
        self.assertIn("manual_goal_use_current_height_ ? odom_pos_(2)", fsm)


if __name__ == "__main__":
    unittest.main()
