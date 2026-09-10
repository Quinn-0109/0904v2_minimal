#!/usr/bin/env python3
from pathlib import Path
import unittest


class MergeTraceResultsTest(unittest.TestCase):
    def test_merge_summary_includes_region_coverage_metrics(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "merge_trace_results.py"
        source = script.read_text(encoding="utf-8")

        self.assertIn("compute_region_metrics", source)
        self.assertIn("coverage_%s_observed_cells", source)


if __name__ == "__main__":
    unittest.main()
