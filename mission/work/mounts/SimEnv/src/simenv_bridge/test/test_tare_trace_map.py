#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import sys
import types
import unittest

import numpy as np


def _install_ros_stubs():
    rospy = types.ModuleType("rospy")
    rospy.get_param = lambda _name, default=None: default
    rospy.Time = types.SimpleNamespace(now=lambda: types.SimpleNamespace(to_sec=lambda: 0.0))
    rospy.loginfo = lambda *args, **kwargs: None
    rospy.on_shutdown = lambda _fn: None
    rospy.Rate = lambda _hz: types.SimpleNamespace(sleep=lambda: None)
    rospy.is_shutdown = lambda: True
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
    sys.modules["sensor_msgs.point_cloud2"].read_points = lambda *_args, **_kwargs: iter([])
    sys.modules["nav_msgs.msg"].Odometry = type("Odometry", (), {})
    sys.modules["geometry_msgs.msg"].PointStamped = type("PointStamped", (), {})
    sys.modules["geometry_msgs.msg"].Twist = type("Twist", (), {})


def _load_module():
    _install_ros_stubs()
    script = Path(__file__).resolve().parents[1] / "scripts" / "tare_trace_map.py"
    spec = importlib.util.spec_from_file_location("tare_trace_map_under_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TareTraceMapTest(unittest.TestCase):
    def test_filter_global_points_discards_layout_xy_and_height_outliers(self):
        mod = _load_module()
        layout = {
            "wall_height": 3.0,
            "floors": [
                {
                    "lobby_bounds": {"x_min": -10.0, "x_max": 10.0, "y_min": 0.0, "y_max": 8.0},
                    "corridor_bounds": {"x_min": -1.0, "x_max": 1.0, "y_min": 8.0, "y_max": 36.0},
                    "rooms": [
                        {"id": "left", "bounds": {"x_min": -9.5, "x_max": -1.1, "y_min": 8.0, "y_max": 36.0}},
                    ],
                }
            ],
        }
        points = np.array(
            [
                [0.0, 2.0, 1.0],
                [-5.0, 20.0, 2.0],
                [0.0, -20.0, 1.0],
                [30.0, 2.0, 1.0],
                [0.0, 2.0, 4.5],
                [0.0, 2.0, -2.0],
            ],
            dtype=np.float32,
        )

        filtered = mod.filter_global_points(points, layout, xy_margin=0.5, z_margin=0.8)

        self.assertEqual(filtered.tolist(), [[0.0, 2.0, 1.0], [-5.0, 20.0, 2.0]])

    def test_raycast_cells_marks_free_cells_before_endpoint(self):
        mod = _load_module()

        free, occ = mod.raycast_xy_cells(
            origin_xy=(0.0, 0.0),
            point_xy=(1.2, 0.0),
            grid_res=0.5,
        )

        self.assertIn((0, 0), free)
        self.assertIn((1, 0), free)
        self.assertEqual(occ, (2, 0))
        self.assertNotIn(occ, free)

    def test_accumulate_occupancy_uses_wall_height_points_as_obstacles(self):
        mod = _load_module()
        points = np.array(
            [
                [1.2, 0.0, 1.0],
                [2.0, 0.0, 2.5],
                [3.0, 0.0, -0.5],
            ],
            dtype=np.float32,
        )
        free, occ = set(), set()

        mod.accumulate_scan_occupancy(
            free_cells=free,
            occupied_cells=occ,
            origin_xy=(0.0, 0.0),
            world_points=points,
            grid_res=0.5,
            z_min=0.15,
            z_max=1.6,
            max_rays=100,
        )

        self.assertEqual(occ, {(2, 0)})
        self.assertIn((0, 0), free)
        self.assertIn((1, 0), free)

    def test_filter_points_to_robot_region_prevents_cross_wall_raycast(self):
        mod = _load_module()
        layout = {
            "floors": [
                {
                    "lobby_bounds": {"x_min": -10.0, "x_max": 10.0, "y_min": 0.0, "y_max": 8.0},
                    "corridor_bounds": {"x_min": -1.0, "x_max": 1.0, "y_min": 8.0, "y_max": 36.0},
                    "rooms": [
                        {"id": "left", "bounds": {"x_min": -9.5, "x_max": -1.1, "y_min": 8.0, "y_max": 36.0}},
                        {"id": "right", "bounds": {"x_min": 1.1, "x_max": 9.5, "y_min": 8.0, "y_max": 36.0}},
                    ],
                }
            ],
        }
        points = np.array(
            [
                [-5.0, 20.0, 1.0],
                [5.0, 20.0, 1.0],
                [0.0, 20.0, 1.0],
            ],
            dtype=np.float32,
        )

        filtered = mod.filter_points_to_robot_region(points, layout, robot_x=-4.0, robot_y=20.0, margin=0.2)

        self.assertEqual(filtered.tolist(), [[-5.0, 20.0, 1.0]])

    def test_filter_points_to_robot_region_returns_empty_when_robot_outside_layout(self):
        mod = _load_module()
        layout = {
            "floors": [
                {
                    "lobby_bounds": {"x_min": -10.0, "x_max": 10.0, "y_min": 0.0, "y_max": 8.0},
                    "corridor_bounds": {"x_min": -1.0, "x_max": 1.0, "y_min": 8.0, "y_max": 36.0},
                    "rooms": [],
                }
            ],
        }
        points = np.array([[0.0, 2.0, 1.0], [0.0, 4.0, 1.0]], dtype=np.float32)

        filtered = mod.filter_points_to_robot_region(points, layout, robot_x=0.0, robot_y=-2.2)

        self.assertEqual(filtered.shape, (0, 3))
        self.assertEqual(filtered.dtype, points.dtype)

    def test_compute_region_metrics_reports_trajectory_and_free_cell_coverage(self):
        mod = _load_module()
        layout = {
            "floors": [
                {
                    "lobby_bounds": {"x_min": 0.0, "x_max": 2.0, "y_min": 0.0, "y_max": 1.0},
                    "corridor_bounds": {"x_min": 0.0, "x_max": 1.0, "y_min": 1.0, "y_max": 3.0},
                    "rooms": [],
                }
            ],
        }
        trajectory = [
            [0.0, 0.1, 0.1, 0.0, 0.0],
            [1.0, 1.9, 0.9, 0.0, 0.0],
        ]
        free_cells = {(0, 0), (1, 0), (2, 0), (3, 0)}
        occupied_cells = {(3, 1)}

        metrics = mod.compute_region_metrics(layout, trajectory, free_cells, occupied_cells, grid_res=0.5)

        self.assertEqual(metrics["lobby"]["trajectory_samples"], 2)
        self.assertGreater(metrics["lobby"]["trajectory_path"], 0.0)
        self.assertAlmostEqual(metrics["lobby"]["trajectory_x_coverage"], 0.9)
        self.assertAlmostEqual(metrics["lobby"]["trajectory_y_coverage"], 0.8)
        self.assertAlmostEqual(metrics["lobby"]["free_cell_coverage"], 0.5)
        self.assertAlmostEqual(metrics["lobby"]["occupied_cell_coverage"], 0.125)


if __name__ == "__main__":
    unittest.main()
