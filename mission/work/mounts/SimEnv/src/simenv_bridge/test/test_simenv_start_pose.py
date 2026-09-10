#!/usr/bin/env python3
from pathlib import Path
import re
import unittest


class SimenvStartPoseTest(unittest.TestCase):
    def test_default_robot_start_is_inside_generated_lobby(self):
        auto_sh = Path(__file__).resolve().parents[3] / "auto.sh"
        text = auto_sh.read_text()

        match = re.search(r'ROBOT_Y="\$\{ROBOT_Y:-(?P<value>[-0-9.]+)\}"', text)
        self.assertIsNotNone(match)
        default_y = float(match.group("value"))

        self.assertGreaterEqual(default_y, 0.0)
        self.assertLessEqual(default_y, 8.0)

    def test_run_script_passes_lobby_start_default_without_rebuild(self):
        run_sh = Path(__file__).resolve().parents[3] / "docker" / "run.sh"
        text = run_sh.read_text()

        self.assertIn('-e ROBOT_Y="${ROBOT_Y:-2.0}"', text)


if __name__ == "__main__":
    unittest.main()
