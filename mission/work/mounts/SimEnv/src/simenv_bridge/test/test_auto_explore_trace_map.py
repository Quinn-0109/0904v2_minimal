#!/usr/bin/env python3

import importlib.util
from pathlib import Path
import sys
import types
import unittest


def _install_ros_stubs():
    rospy = types.ModuleType("rospy")
    rospy.get_param = lambda _name, default=None: default
    rospy.Time = types.SimpleNamespace(now=lambda: types.SimpleNamespace(to_sec=lambda: 0.0))
    rospy.Duration = lambda value: value
    rospy.loginfo = lambda *args, **kwargs: None
    rospy.logwarn = lambda *args, **kwargs: None
    sys.modules["rospy"] = rospy

    for name in [
        "sensor_msgs",
        "sensor_msgs.msg",
        "sensor_msgs.point_cloud2",
        "nav_msgs",
        "nav_msgs.msg",
        "geometry_msgs",
        "geometry_msgs.msg",
    ]:
        sys.modules[name] = types.ModuleType(name)
    sys.modules["sensor_msgs.msg"].PointCloud2 = type("PointCloud2", (), {})
    sys.modules["nav_msgs.msg"].Odometry = type("Odometry", (), {})
    sys.modules["geometry_msgs.msg"].Twist = type("Twist", (), {})


def _load_module():
    _install_ros_stubs()
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "auto_explore_trace_map.py"
    spec = importlib.util.spec_from_file_location("auto_explore_trace_map_under_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AutoExploreTraceMapTest(unittest.TestCase):
    def test_default_speed_is_one_meter_per_second(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "auto_explore_trace_map.py"
        text = script.read_text()

        self.assertIn('rospy.get_param("~v_max", 1.0)', text)

    def test_plan_waypoints_cover_layout_regions(self):
        mod = _load_module()
        layout = {
            "floors": [
                {
                    "lobby_bounds": {"x_min": -10.0, "x_max": 10.0, "y_min": 0.0, "y_max": 8.0},
                    "corridor_bounds": {"x_min": -1.0, "x_max": 1.0, "y_min": 8.0, "y_max": 36.0},
                    "rooms": [
                        {
                            "id": "left",
                            "bounds": {"x_min": -9.5, "x_max": -1.1, "y_min": 8.0, "y_max": 36.0},
                            "goal_pose": [-4.8, 22.0, 0, 0, 0, 0],
                            "door_pose": [-1.1, 22.0, 1.2, 0, 0, 0],
                        },
                        {
                            "id": "right",
                            "bounds": {"x_min": 1.1, "x_max": 9.5, "y_min": 8.0, "y_max": 36.0},
                            "goal_pose": [4.8, 22.0, 0, 0, 0, 0],
                            "door_pose": [1.1, 22.0, 1.2, 0, 0, 3.14],
                        },
                    ],
                }
            ]
        }

        waypoints = mod.plan_exploration_waypoints(layout, y_step=7.0)
        labels = [wp.label for wp in waypoints]

        self.assertIn("lobby_center", labels)
        self.assertTrue(any(label.startswith("corridor_") for label in labels))
        self.assertTrue(any(label.startswith("left_") for label in labels))
        self.assertTrue(any(label.startswith("right_") for label in labels))
        self.assertGreater(len(waypoints), 8)

    def test_compact_waypoints_visit_each_room_goal_with_fewer_targets(self):
        mod = _load_module()
        layout = {
            "floors": [
                {
                    "lobby_bounds": {"x_min": -10.0, "x_max": 10.0, "y_min": 0.0, "y_max": 8.0},
                    "corridor_bounds": {"x_min": -1.0, "x_max": 1.0, "y_min": 8.0, "y_max": 36.0},
                    "rooms": [
                        {
                            "id": "left",
                            "bounds": {"x_min": -9.5, "x_max": -1.1, "y_min": 8.0, "y_max": 36.0},
                            "goal_pose": [-4.8, 22.0, 0, 0, 0, 0],
                            "door_pose": [-1.1, 22.0, 1.2, 0, 0, 0],
                        },
                        {
                            "id": "right",
                            "bounds": {"x_min": 1.1, "x_max": 9.5, "y_min": 8.0, "y_max": 36.0},
                            "goal_pose": [4.8, 28.0, 0, 0, 0, 0],
                            "door_pose": [1.1, 28.0, 1.2, 0, 0, 3.14],
                        },
                    ],
                }
            ]
        }

        dense = mod.plan_exploration_waypoints(layout, y_step=7.0)
        compact = mod.plan_exploration_waypoints(layout, y_step=7.0, route_mode="compact")
        labels = [wp.label for wp in compact]

        self.assertLess(len(compact), len(dense))
        self.assertIn("left_goal", labels)
        self.assertIn("right_goal", labels)
        self.assertTrue(any(label == "corridor_end" for label in labels))

    def test_compute_cmd_turns_toward_target_and_stops_near_goal(self):
        mod = _load_module()
        far_cmd, done = mod.compute_cmd_to_waypoint(
            pose=(0.0, 0.0, 0.0),
            target=mod.Waypoint(2.0, 1.0, "target"),
            v_max=0.5,
            w_max=0.8,
            goal_tol=0.3,
            slow_radius=1.0,
        )
        self.assertFalse(done)
        self.assertGreater(far_cmd.vx, 0.0)
        self.assertGreater(far_cmd.wz, 0.0)

        near_cmd, done = mod.compute_cmd_to_waypoint(
            pose=(0.0, 0.0, 0.0),
            target=mod.Waypoint(0.1, 0.1, "target"),
            v_max=0.5,
            w_max=0.8,
            goal_tol=0.3,
            slow_radius=1.0,
        )
        self.assertTrue(done)
        self.assertEqual(near_cmd.vx, 0.0)
        self.assertEqual(near_cmd.wz, 0.0)

    def test_world_frame_command_points_directly_at_target(self):
        mod = _load_module()
        cmd, done = mod.compute_cmd_to_waypoint(
            pose=(0.0, -2.0, 1.5708),
            target=mod.Waypoint(0.0, 1.0, "entrance"),
            v_max=0.5,
            w_max=0.8,
            goal_tol=0.3,
            slow_radius=1.0,
            command_frame="world",
        )

        self.assertFalse(done)
        self.assertAlmostEqual(cmd.vx, 0.0, places=3)
        self.assertGreater(cmd.vy, 0.0)

    def test_filter_global_points_discards_height_outliers(self):
        mod = _load_module()
        layout = {"wall_height": 3.0}
        points = mod.np.array(
            [
                [0.0, 0.0, -0.2],
                [1.0, 1.0, 2.0],
                [2.0, 2.0, 12.0],
                [3.0, 3.0, -8.0],
            ],
            dtype=mod.np.float32,
        )

        filtered = mod.filter_global_points(points, layout, z_margin=0.8)

        self.assertEqual(len(filtered), 2)
        self.assertTrue((filtered[:, 2] <= 3.8).all())
        self.assertTrue((filtered[:, 2] >= -0.8).all())

    def test_extract_xyz_reads_livox_custom_msg_points(self):
        mod = _load_module()
        points = [
            types.SimpleNamespace(x=1.0, y=2.0, z=3.0),
            types.SimpleNamespace(x=-1.0, y=-2.0, z=-3.0),
        ]
        msg = types.SimpleNamespace(points=points)

        xyz = mod.extract_xyz_points(msg)

        self.assertEqual(xyz.shape, (2, 3))
        self.assertEqual(xyz[0].tolist(), [1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
