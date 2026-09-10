#!/usr/bin/env python3
"""ROS-free regression tests for hierarchical exploration decisions."""

import os
import sys
import unittest

import numpy as np

SCRIPT_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "scripts"))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from baseline_planning_core import OccupancyGrid2D
from exploration_graph import (
    ExplorationGraph, ExplorationNode, FrontierCluster,
    FrontierClusterDetector)
from hierarchical_explorer import HierarchicalGoalSelector, SelectorConfig
from planner_interface import AStarPlanner, FarPlannerInterface


def grid(data, resolution=0.25):
    return OccupancyGrid2D(
        np.asarray(data, dtype=np.int16), resolution, 0.0, 0.0)


class HierarchicalExplorationTest(unittest.TestCase):
    def test_frontier_points_are_clustered(self):
        data = np.full((40, 40), -1, dtype=np.int16)
        data[10:30, 8:25] = 0
        detector = FrontierClusterDetector(
            minimum_cluster_cells=4, merge_distance=1.0)
        clusters = detector.detect(grid(data), (3.0, 4.0))
        self.assertGreaterEqual(len(clusters), 1)
        self.assertTrue(all(item.size >= 4 for item in clusters))
        self.assertTrue(any(item.reachable for item in clusters))
        planner = AStarPlanner(clearance=0.30)
        self.assertTrue(all(
            planner.plan((3.0, 4.0), item.goal, grid(data)).success
            for item in clusters if item.reachable))

    def test_region_identity_survives_frontier_motion(self):
        data = np.full((80, 100), 100, dtype=np.int16)
        data[10:65, 10:45] = 0
        map_grid = grid(data)
        graph = ExplorationGraph(
            region_core_clearance=0.5,
            minimum_region_core_cells=20,
            region_match_distance=4.0)
        first = FrontierCluster(
            "f1", (10.5, 8.0), (10.0, 8.0), 20, 8.0, 5.0,
            True, 4.0)
        graph.update(map_grid, [first], (5.0, 8.0), 1.0)
        region_ids = [
            item.id for item in graph.nodes.values()
            if item.type == "region"]
        self.assertEqual(len(region_ids), 1)
        region_id = region_ids[0]
        graph.mark_region(
            region_id, 0.92, 2.0, entered=True, exited=True)
        second = FrontierCluster(
            "f2", (3.0, 14.0), (3.5, 14.0), 18, 6.0, 4.5,
            True, 8.0)
        graph.update(map_grid, [second], (5.0, 8.0), 3.0)
        persistent = graph.nodes[region_id]
        self.assertEqual(persistent.type, "visited")
        self.assertEqual(persistent.visited_count, 1)
        self.assertGreaterEqual(persistent.coverage_ratio, 0.92)

    def test_selector_penalizes_revisit_and_uses_astar_cost(self):
        data = np.full((50, 80), -1, dtype=np.int16)
        data[4:46, 4:76] = 0
        map_grid = grid(data)
        graph = ExplorationGraph()
        graph.nodes = {
            "region_near": ExplorationNode(
                "region_near", "region", (3.0, 3.0),
                information_gain=5.0, visited_count=2,
                goal=(3.0, 3.0)),
            "region_far": ExplorationNode(
                "region_far", "region", (8.0, 3.0),
                information_gain=12.0, visited_count=0,
                goal=(8.0, 3.0)),
        }
        selector = HierarchicalGoalSelector(
            AStarPlanner(clearance=0.0),
            SelectorConfig(minimum_information_gain=0.1))
        decision = selector.select(
            graph, (1.0, 3.0), map_grid, 500.0)
        self.assertEqual(decision.kind, "enter_region")
        self.assertEqual(decision.region_id, "region_far")
        self.assertTrue(decision.path_result.success)

    def test_only_physical_region_visit_can_mark_visited(self):
        graph = ExplorationGraph()
        graph.nodes["region_001"] = ExplorationNode(
            "region_001", "region", (2.0, 2.0),
            coverage_ratio=0.0, visited=False)
        graph.mark_region(
            "region_001", 0.95, 1.0, entered=False, exited=True,
            coverage_eligible=False)
        self.assertFalse(graph.nodes["region_001"].visited)
        graph.mark_region(
            "region_001", 0.91, 2.0, entered=True, exited=True)
        self.assertTrue(graph.nodes["region_001"].visited)
        self.assertEqual(graph.nodes["region_001"].type, "visited")

    def test_far_boundary_currently_delegates_to_astar(self):
        map_grid = grid(np.zeros((30, 30), dtype=np.int16))
        result = FarPlannerInterface(
            AStarPlanner(clearance=0.0)).plan(
                (1.0, 1.0), (5.0, 5.0), map_grid)
        self.assertTrue(result.success)
        self.assertEqual(result.metadata["interface"], "far_compatible")
        self.assertEqual(result.metadata["active_backend"], "astar")


if __name__ == "__main__":
    unittest.main()
