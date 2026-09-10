from pathlib import Path
import unittest
import xml.etree.ElementTree as ET


class ExploreLaunchScanSourceTest(unittest.TestCase):
    def test_default_scan_input_uses_native_ultra_fusion_livox_pointcloud_source(self):
        launch = Path(__file__).resolve().parents[1] / "launch" / "explore_simenv.launch"
        root = ET.parse(launch).getroot()
        scan_arg = next(arg for arg in root.findall("arg") if arg.attrib.get("name") == "scan_input")

        self.assertEqual(scan_arg.attrib.get("default"), "/livox/Pointcloud2")

    def test_scan_to_uf_defaults_to_native_livox_pointcloud_source(self):
        launch = Path(__file__).resolve().parents[1] / "launch" / "explore_simenv.launch"
        root = ET.parse(launch).getroot()
        scan_node = next(node for node in root.findall("node") if node.attrib.get("name") == "scan_to_uf")
        scan_input = next(param for param in scan_node.findall("param") if param.attrib.get("name") == "input")
        prefer_custom = next(param for param in scan_node.findall("param") if param.attrib.get("name") == "prefer_custom")

        self.assertEqual(scan_input.attrib.get("value"), "/livox/Pointcloud2")
        self.assertEqual(prefer_custom.attrib.get("value"), "false")

    def test_registered_scan_range_matches_long_corridor_simenv_tare_config(self):
        launch = Path(__file__).resolve().parents[1] / "launch" / "explore_simenv.launch"
        root = ET.parse(launch).getroot()
        scan_range_arg = next(arg for arg in root.findall("arg") if arg.attrib.get("name") == "scan_max_range")
        scan_node = next(node for node in root.findall("node") if node.attrib.get("name") == "scan_to_map")
        max_range = next(param for param in scan_node.findall("param") if param.attrib.get("name") == "max_range")

        self.assertEqual(scan_range_arg.attrib.get("default"), "12.0")
        self.assertEqual(max_range.attrib.get("value"), "$(arg scan_max_range)")


if __name__ == "__main__":
    unittest.main()
