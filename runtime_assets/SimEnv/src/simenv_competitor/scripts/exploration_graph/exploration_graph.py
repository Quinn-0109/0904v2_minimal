"""Persistent topological exploration graph built from online map evidence."""

from collections import deque
from dataclasses import dataclass, field
import heapq
import math
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .frontier_cluster import FrontierCluster

Cell = Tuple[int, int]
Point = Tuple[float, float]


@dataclass
class ExplorationNode:
    id: str
    type: str
    position: Point
    size: float = 0.0
    information_gain: float = 0.0
    coverage_ratio: float = 0.0
    visited_count: int = 0
    last_visit_time: float = 0.0
    reachable: bool = True
    visited: bool = False
    frontier_ids: List[str] = field(default_factory=list)
    goal: Optional[Point] = None
    metadata: dict = field(default_factory=dict, repr=False)

    def to_dict(self, include_metadata: bool = False) -> dict:
        payload = {
            "id": self.id,
            "type": self.type,
            "position": list(self.position),
            "size": self.size,
            "information_gain": self.information_gain,
            "coverage_ratio": self.coverage_ratio,
            "visited_count": self.visited_count,
            "last_visit_time": self.last_visit_time,
            "reachable": self.reachable,
            "visited": self.visited,
            "frontier_ids": list(self.frontier_ids),
            "goal": list(self.goal) if self.goal is not None else None,
        }
        if include_metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "ExplorationNode":
        goal = payload.get("goal")
        return cls(
            id=str(payload["id"]),
            type=str(payload["type"]),
            position=tuple(map(float, payload["position"][:2])),
            size=float(payload.get("size", 0.0)),
            information_gain=float(payload.get("information_gain", 0.0)),
            coverage_ratio=float(payload.get("coverage_ratio", 0.0)),
            visited_count=int(payload.get("visited_count", 0)),
            last_visit_time=float(payload.get("last_visit_time", 0.0)),
            reachable=bool(payload.get("reachable", True)),
            visited=bool(payload.get("visited", False)),
            frontier_ids=list(payload.get("frontier_ids", [])),
            goal=(tuple(map(float, goal[:2])) if goal is not None else None),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass
class ExplorationEdge:
    from_id: str
    to_id: str
    distance: float
    travel_cost: float
    connectivity: float

    def to_dict(self) -> dict:
        return {
            "from_id": self.from_id,
            "to_id": self.to_id,
            "distance": self.distance,
            "travel_cost": self.travel_cost,
            "connectivity": self.connectivity,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ExplorationEdge":
        return cls(
            from_id=str(payload["from_id"]),
            to_id=str(payload["to_id"]),
            distance=float(payload.get("distance", 0.0)),
            travel_cost=float(payload.get("travel_cost", 0.0)),
            connectivity=float(payload.get("connectivity", 1.0)),
        )


class ExplorationGraph:
    NODE_TYPES = {"frontier_cluster", "region", "corridor", "visited"}

    def __init__(self, region_core_clearance: float = 0.75,
                 minimum_region_core_cells: int = 20,
                 region_association_radius: float = 3.0,
                 region_match_distance: float = 5.0,
                 graph_connection_distance: float = 15.0):
        self.region_core_clearance = float(region_core_clearance)
        self.minimum_region_core_cells = int(minimum_region_core_cells)
        self.region_association_radius = float(region_association_radius)
        self.region_match_distance = float(region_match_distance)
        self.graph_connection_distance = float(graph_connection_distance)
        self.nodes: Dict[str, ExplorationNode] = {}
        self.edges: List[ExplorationEdge] = []
        self.update_sequence = 0
        self._region_sequence = 0

    @staticmethod
    def _neighbors4(cell: Cell) -> Iterable[Cell]:
        x, y = cell
        return ((x + 1, y), (x - 1, y),
                (x, y + 1), (x, y - 1))

    @staticmethod
    def _neighbors8(cell: Cell) -> Iterable[Cell]:
        x, y = cell
        return (
            (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1),
            (x + 1, y + 1), (x + 1, y - 1),
            (x - 1, y + 1), (x - 1, y - 1),
        )

    @staticmethod
    def _inside(grid, cell: Cell) -> bool:
        return 0 <= cell[0] < grid.width and 0 <= cell[1] < grid.height

    def _clearance_cells(self, grid) -> np.ndarray:
        """Obstacle/unknown clearance using an 8-neighbour distance transform."""
        blocked = grid.data != 0
        distance = np.full(blocked.shape, np.inf, dtype=np.float32)
        queue = []
        for y, x in np.argwhere(blocked):
            cell = int(x), int(y)
            distance[cell[1], cell[0]] = 0.0
            heapq.heappush(queue, (0.0, cell))
        diagonal = math.sqrt(2.0)
        while queue:
            base, cell = heapq.heappop(queue)
            if base > float(distance[cell[1], cell[0]]) + 1e-6:
                continue
            for nxt in self._neighbors8(cell):
                if not self._inside(grid, nxt):
                    continue
                cost = diagonal if (
                    nxt[0] != cell[0] and nxt[1] != cell[1]) else 1.0
                candidate = base + cost
                if candidate + 1e-6 >= distance[nxt[1], nxt[0]]:
                    continue
                distance[nxt[1], nxt[0]] = candidate
                heapq.heappush(queue, (candidate, nxt))
        return distance

    def _region_cores(self, grid
                      ) -> Tuple[np.ndarray, Dict[int, List[Cell]]]:
        clearance = self._clearance_cells(grid)
        core = np.logical_and(
            grid.data == 0,
            clearance * float(grid.resolution) >=
            self.region_core_clearance)
        labels = np.full(core.shape, -1, dtype=np.int32)
        components: Dict[int, List[Cell]] = {}
        next_label = 0
        for y, x in np.argwhere(core):
            seed = int(x), int(y)
            if labels[seed[1], seed[0]] >= 0:
                continue
            queue, cells = deque([seed]), [seed]
            labels[seed[1], seed[0]] = next_label
            while queue:
                cell = queue.popleft()
                for nxt in self._neighbors8(cell):
                    if (not self._inside(grid, nxt) or
                            not core[nxt[1], nxt[0]] or
                            labels[nxt[1], nxt[0]] >= 0):
                        continue
                    labels[nxt[1], nxt[0]] = next_label
                    cells.append(nxt)
                    queue.append(nxt)
            if len(cells) >= self.minimum_region_core_cells:
                components[next_label] = cells
                next_label += 1
            else:
                for cell in cells:
                    labels[cell[1], cell[0]] = -1
        return labels, components

    def _cluster_core_label(self, grid, labels: np.ndarray,
                            cluster: FrontierCluster) -> Optional[int]:
        goal_cell = grid.world_to_cell(cluster.goal)
        if goal_cell is None:
            return None
        radius = max(1, int(math.ceil(
            self.region_association_radius / grid.resolution)))
        best = None
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                cell = goal_cell[0] + dx, goal_cell[1] + dy
                if not self._inside(grid, cell):
                    continue
                label = int(labels[cell[1], cell[0]])
                if label < 0:
                    continue
                distance = math.hypot(dx, dy) * grid.resolution
                if best is None or distance < best[0]:
                    best = distance, label
        return None if best is None else best[1]

    @staticmethod
    def _sample_component(grid, cells: List[Cell],
                          maximum: int = 64) -> List[Point]:
        if not cells:
            return []
        stride = max(1, len(cells) // maximum)
        return [
            tuple(map(float, grid.cell_to_world(cell)))
            for cell in cells[::stride][:maximum]
        ]

    def _match_region(self, grid, label: Optional[int],
                      position: Point) -> Optional[ExplorationNode]:
        candidates = [
            node for node in self.nodes.values()
            if node.type in ("region", "visited")]
        if label is not None:
            for node in candidates:
                for point in node.metadata.get("support_points", []):
                    cell = grid.world_to_cell(point)
                    if cell is not None and int(
                            self._current_labels[cell[1], cell[0]]) == label:
                        return node
        nearby = [
            node for node in candidates
            if math.hypot(node.position[0] - position[0],
                          node.position[1] - position[1]) <=
            self.region_match_distance]
        return min(
            nearby,
            key=lambda node: math.hypot(
                node.position[0] - position[0],
                node.position[1] - position[1]),
            default=None)

    def _new_region_id(self) -> str:
        self._region_sequence += 1
        return "region_{:03d}".format(self._region_sequence)

    def _update_regions(self, grid, clusters: List[FrontierCluster],
                        labels: np.ndarray,
                        components: Dict[int, List[Cell]]) -> Dict[str, str]:
        grouped: Dict[object, List[FrontierCluster]] = {}
        for cluster in clusters:
            label = self._cluster_core_label(grid, labels, cluster)
            cluster.region_hint = label
            key = ("core", label) if label is not None else (
                "orphan",
                int(round(cluster.center[0] /
                          max(0.5, self.region_match_distance))),
                int(round(cluster.center[1] /
                          max(0.5, self.region_match_distance))))
            grouped.setdefault(key, []).append(cluster)
        associations: Dict[str, str] = {}
        touched: Set[str] = set()
        for key, members in grouped.items():
            label = key[1] if key[0] == "core" else None
            if label is not None and label in components:
                points = [grid.cell_to_world(cell)
                          for cell in components[label]]
                position = (
                    sum(point[0] for point in points) / len(points),
                    sum(point[1] for point in points) / len(points))
                support = self._sample_component(
                    grid, components[label])
                size = len(components[label]) * grid.resolution ** 2
            else:
                total = max(1, sum(item.size for item in members))
                position = (
                    sum(item.center[0] * item.size for item in members) / total,
                    sum(item.center[1] * item.size for item in members) / total)
                support = [item.goal for item in members]
                size = sum(item.size for item in members) * \
                    grid.resolution ** 2
            node = self._match_region(grid, label, position)
            if node is None:
                node = ExplorationNode(
                    id=self._new_region_id(), type="region",
                    position=(float(position[0]), float(position[1])))
                self.nodes[node.id] = node
            touched.add(node.id)
            previous_weight = max(1.0, float(node.metadata.get(
                "observation_count", 0.0)))
            node.position = (
                (node.position[0] * previous_weight + position[0]) /
                (previous_weight + 1.0),
                (node.position[1] * previous_weight + position[1]) /
                (previous_weight + 1.0))
            node.metadata["observation_count"] = previous_weight + 1.0
            node.metadata["support_points"] = [
                list(point) for point in support]
            node.size = max(node.size, float(size))
            node.information_gain = sum(
                item.unknown_area for item in members)
            node.reachable = any(item.reachable for item in members)
            node.frontier_ids = [item.id for item in members]
            reachable_members = [item for item in members if item.reachable]
            selected = max(
                reachable_members or members,
                key=lambda item: (
                    item.unknown_area, item.boundary_length))
            node.goal = selected.goal
            node.metadata["last_observed_update"] = self.update_sequence
            for item in members:
                associations[item.id] = node.id
        for node in self.nodes.values():
            if node.type in ("region", "visited") and node.id not in touched:
                node.frontier_ids = []
                node.information_gain = 0.0
                node.reachable = False
        return associations

    def _rebuild_edges(self, associations: Dict[str, str]) -> None:
        self.edges = []
        frontier_nodes = [
            node for node in self.nodes.values()
            if node.type == "frontier_cluster"]
        for frontier in frontier_nodes:
            region_id = associations.get(frontier.id)
            if region_id is None:
                continue
            distance = math.hypot(
                frontier.position[0] - self.nodes[region_id].position[0],
                frontier.position[1] - self.nodes[region_id].position[1])
            self.edges.append(ExplorationEdge(
                region_id, frontier.id, distance, distance, 1.0))
        regions = [
            node for node in self.nodes.values()
            if node.type in ("region", "visited") and node.reachable]
        for index, first in enumerate(regions):
            neighbours = sorted(
                (second for second in regions[index + 1:]
                 if math.hypot(
                     first.position[0] - second.position[0],
                     first.position[1] - second.position[1]) <=
                 self.graph_connection_distance),
                key=lambda second: math.hypot(
                    first.position[0] - second.position[0],
                    first.position[1] - second.position[1]))[:3]
            for second in neighbours:
                distance = math.hypot(
                    first.position[0] - second.position[0],
                    first.position[1] - second.position[1])
                pair = tuple(sorted((first.id, second.id)))
                corridor_id = "corridor_{}_{}".format(*pair)
                corridor = self.nodes.get(corridor_id)
                if corridor is None:
                    corridor = ExplorationNode(
                        id=corridor_id, type="corridor",
                        position=(
                            (first.position[0] + second.position[0]) * 0.5,
                            (first.position[1] + second.position[1]) * 0.5),
                        reachable=True)
                    self.nodes[corridor_id] = corridor
                self.edges.extend([
                    ExplorationEdge(
                        first.id, corridor.id, distance * 0.5,
                        distance * 0.5, 1.0),
                    ExplorationEdge(
                        corridor.id, second.id, distance * 0.5,
                        distance * 0.5, 1.0),
                ])

    def update(self, grid, clusters: List[FrontierCluster],
               robot_pose: Sequence[float], now: float) -> dict:
        del robot_pose, now
        self.update_sequence += 1
        self._current_labels, components = self._region_cores(grid)
        for node_id in [
            node.id for node in self.nodes.values()
            if node.type in ("frontier_cluster", "corridor")
        ]:
            self.nodes.pop(node_id, None)
        for cluster in clusters:
            self.nodes[cluster.id] = ExplorationNode(
                id=cluster.id,
                type="frontier_cluster",
                position=cluster.center,
                size=float(cluster.size),
                information_gain=cluster.unknown_area,
                reachable=cluster.reachable,
                goal=cluster.goal,
                metadata={
                    "boundary_length": cluster.boundary_length,
                    "distance_to_robot": cluster.distance_to_robot,
                })
        associations = self._update_regions(
            grid, clusters, self._current_labels, components)
        self._rebuild_edges(associations)
        return {
            "region_count": sum(
                node.type in ("region", "visited")
                for node in self.nodes.values()),
            "frontier_count": len(clusters),
            "edge_count": len(self.edges),
            "unknown_ratio": self.unknown_ratio(grid),
        }

    @staticmethod
    def unknown_ratio(grid) -> float:
        domain = int(np.count_nonzero(grid.data <= 0))
        return (float(np.count_nonzero(grid.data < 0)) / domain
                if domain else 1.0)

    def mark_region(self, region_id: str, coverage_ratio: float,
                    now: float, entered: bool = False,
                    exited: bool = False,
                    coverage_threshold: float = 0.90,
                    coverage_eligible: bool = True) -> None:
        node = self.nodes.get(region_id)
        if node is None or node.type not in ("region", "visited"):
            return
        node.coverage_ratio = max(
            node.coverage_ratio, float(coverage_ratio))
        if entered:
            node.visited_count += 1
            node.last_visit_time = float(now)
        if exited:
            node.last_visit_time = float(now)
        if (coverage_eligible and
                node.coverage_ratio >= float(coverage_threshold)):
            node.visited = True
            node.type = "visited"

    def neighbours(self, node_id: str) -> List[str]:
        output = []
        for edge in self.edges:
            if edge.from_id == node_id:
                output.append(edge.to_id)
            elif edge.to_id == node_id:
                output.append(edge.from_id)
        return output

    def to_dict(self, include_metadata: bool = False) -> dict:
        return {
            "schema": "simenv_exploration_graph_v1",
            "update_sequence": self.update_sequence,
            "nodes": [
                node.to_dict(include_metadata)
                for node in sorted(
                    self.nodes.values(), key=lambda item: item.id)
            ],
            "edges": [edge.to_dict() for edge in self.edges],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ExplorationGraph":
        graph = cls()
        graph.update_sequence = int(payload.get("update_sequence", 0))
        graph.nodes = {
            item["id"]: ExplorationNode.from_dict(item)
            for item in payload.get("nodes", [])
        }
        graph.edges = [
            ExplorationEdge.from_dict(item)
            for item in payload.get("edges", [])
        ]
        region_ids = [
            int(node_id.split("_")[-1])
            for node_id in graph.nodes
            if node_id.startswith("region_") and
            node_id.split("_")[-1].isdigit()
        ]
        graph._region_sequence = max(region_ids, default=0)
        return graph
