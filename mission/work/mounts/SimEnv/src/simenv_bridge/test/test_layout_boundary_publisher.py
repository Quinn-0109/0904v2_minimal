import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest


def _load_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "layout_boundary_publisher.py"
    spec = importlib.util.spec_from_file_location("layout_boundary_publisher", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LayoutBoundaryPublisherTest(unittest.TestCase):
    def test_compute_outer_bounds_includes_lobby_corridor_rooms_and_elevator(self):
        module = _load_module()
        metadata = {
            "floors": [
                {
                    "lobby_bounds": {"x_min": -10.0, "x_max": 10.0, "y_min": 0.0, "y_max": 7.85},
                    "corridor_bounds": {"x_min": -1.1, "x_max": 1.1, "y_min": 7.85, "y_max": 35.91},
                    "elevator_bounds": {"x_min": 1.65, "x_max": 4.05, "y_min": 1.25, "y_max": 3.95},
                    "rooms": [
                        {"bounds": {"x_min": -9.5, "x_max": -1.1, "y_min": 7.85, "y_max": 35.91}},
                        {"bounds": {"x_min": 1.1, "x_max": 9.5, "y_min": 7.85, "y_max": 35.91}},
                    ],
                }
            ]
        }

        self.assertEqual(module.compute_outer_bounds(metadata, margin=0.5), (-10.5, 10.5, -0.5, 36.41))

    def test_compute_nogo_bounds_marks_elevator_as_non_navigable(self):
        module = _load_module()
        metadata = {
            "floors": [
                {
                    "elevator_bounds": {"x_min": 1.65, "x_max": 4.05, "y_min": 1.25, "y_max": 3.95},
                }
            ]
        }

        self.assertEqual(module.compute_nogo_bounds(metadata, margin=0.2), [(1.45, 4.25, 1.05, 4.15)])

    def test_make_nogo_polygon_uses_z_as_polygon_id_for_tare(self):
        module = _load_module()
        geometry_msgs = types.ModuleType("geometry_msgs")
        geometry_msgs_msg = types.ModuleType("geometry_msgs.msg")

        class Point32:
            def __init__(self, x=0.0, y=0.0, z=0.0):
                self.x = x
                self.y = y
                self.z = z

        class PolygonStamped:
            def __init__(self):
                self.header = types.SimpleNamespace(frame_id="")
                self.polygon = types.SimpleNamespace(points=[])

        geometry_msgs_msg.Point32 = Point32
        geometry_msgs_msg.PolygonStamped = PolygonStamped
        sys.modules["geometry_msgs"] = geometry_msgs
        sys.modules["geometry_msgs.msg"] = geometry_msgs_msg

        polygon = module.make_nogo_polygon([(1.45, 4.25, 1.05, 4.15)], "map")

        self.assertEqual(polygon.header.frame_id, "map")
        self.assertEqual([(p.x, p.y, p.z) for p in polygon.polygon.points], [
            (1.45, 1.05, 0.0),
            (4.25, 1.05, 0.0),
            (4.25, 4.15, 0.0),
            (1.45, 4.15, 0.0),
        ])

    def test_load_metadata_reads_json_file(self):
        module = _load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata_path = Path(temp_dir) / "layout_metadata.json"
            metadata_path.write_text(json.dumps({"floors": []}), encoding="utf-8")

            self.assertEqual(module.load_metadata(str(metadata_path)), {"floors": []})


if __name__ == "__main__":
    unittest.main()
