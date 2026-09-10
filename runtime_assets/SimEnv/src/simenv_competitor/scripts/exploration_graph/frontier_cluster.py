"""Frontier-cell clustering for hierarchical exploration."""

from collections import deque
from dataclasses import asdict, dataclass, field
import math
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

Cell = Tuple[int, int]
Point = Tuple[float, float]


@dataclass
class FrontierCluster:
    id: str
    center: Point
    goal: Point
    size: int
    unknown_area: float
    boundary_length: float
    reachable: bool
    distance_to_robot: float
    cells: List[Cell] = field(default_factory=list, repr=False)
    free_cells: List[Cell] = field(default_factory=list, repr=False)
    region_hint: Optional[int] = None

    def to_dict(self, include_cells: bool = False) -> dict:
        payload = asdict(self)
        payload["center"] = list(self.center)
        payload["goal"] = list(self.goal)
        if not include_cells:
            payload.pop("cells", None)
            payload.pop("free_cells", None)
        else:
            payload["cells"] = [list(cell) for cell in self.cells]
            payload["free_cells"] = [list(cell) for cell in self.free_cells]
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "FrontierCluster":
        return cls(
            id=str(payload["id"]),
            center=tuple(map(float, payload["center"][:2])),
            goal=tuple(map(float, payload.get(
                "goal", payload["center"])[:2])),
            size=int(payload.get("size", 0)),
            unknown_area=float(payload.get("unknown_area", 0.0)),
            boundary_length=float(payload.get("boundary_length", 0.0)),
            reachable=bool(payload.get("reachable", False)),
            distance_to_robot=float(payload.get(
                "distance_to_robot", float("inf"))),
            cells=[tuple(map(int, cell[:2]))
                   for cell in payload.get("cells", [])],
            free_cells=[tuple(map(int, cell[:2]))
                        for cell in payload.get("free_cells", [])],
            region_hint=payload.get("region_hint"),
        )


class FrontierClusterDetector:
    def __init__(self, minimum_cluster_cells: int = 8,
                 merge_distance: float = 0.75,
                 unknown_area_radius: float = 2.0,
                 merge_connectivity_radius: float = 2.5,
                 goal_search_radius: float = 1.5,
                 goal_clearance: float = 0.35):
        if minimum_cluster_cells < 1 or merge_distance < 0.0:
            raise ValueError("invalid frontier clustering configuration")
        self.minimum_cluster_cells = int(minimum_cluster_cells)
        self.merge_distance = float(merge_distance)
        self.unknown_area_radius = float(unknown_area_radius)
        self.merge_connectivity_radius = float(
            merge_connectivity_radius)
        self.goal_search_radius = float(goal_search_radius)
        self.goal_clearance = float(goal_clearance)
        self.update_sequence = 0

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

    def _reachable_free(self, grid, robot_pose: Sequence[float]) -> Set[Cell]:
        start = grid.world_to_cell(
            (float(robot_pose[0]), float(robot_pose[1])))
        if start is None:
            return set()
        if int(grid.data[start[1], start[0]]) != 0:
            candidates = []
            for radius in range(1, 6):
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        cell = start[0] + dx, start[1] + dy
                        if (self._inside(grid, cell) and
                                int(grid.data[cell[1], cell[0]]) == 0):
                            candidates.append(cell)
                if candidates:
                    start = min(
                        candidates,
                        key=lambda item: math.hypot(
                            item[0] - start[0], item[1] - start[1]))
                    break
        if int(grid.data[start[1], start[0]]) != 0:
            return set()
        queue, visited = deque([start]), {start}
        while queue:
            cell = queue.popleft()
            for nxt in self._neighbors8(cell):
                if (nxt in visited or not self._inside(grid, nxt) or
                        int(grid.data[nxt[1], nxt[0]]) != 0):
                    continue
                visited.add(nxt)
                queue.append(nxt)
        return visited

    def _unknown_area(self, grid, cells: List[Cell]) -> float:
        if not cells:
            return 0.0
        radius = max(
            1, int(math.ceil(self.unknown_area_radius / grid.resolution)))
        xs = [cell[0] for cell in cells]
        ys = [cell[1] for cell in cells]
        x0, x1 = max(0, min(xs) - radius), min(grid.width, max(xs) + radius + 1)
        y0, y1 = max(0, min(ys) - radius), min(grid.height, max(ys) + radius + 1)
        unknown = int(np.count_nonzero(grid.data[y0:y1, x0:x1] < 0))
        return unknown * grid.resolution * grid.resolution

    def _build_cluster(self, grid, cells: List[Cell],
                       reachable_free: Set[Cell],
                       robot_pose: Sequence[float], index: int
                       ) -> Optional[FrontierCluster]:
        free_neighbors: Set[Cell] = set()
        for cell in cells:
            for neighbor in self._neighbors8(cell):
                if (self._inside(grid, neighbor) and
                        int(grid.data[neighbor[1], neighbor[0]]) == 0):
                    free_neighbors.add(neighbor)
        frontier_points = [grid.cell_to_world(cell) for cell in cells]
        center = (
            sum(point[0] for point in frontier_points) / len(frontier_points),
            sum(point[1] for point in frontier_points) / len(frontier_points))
        center_cell = grid.world_to_cell(center)
        search_cells = set()
        if center_cell is not None:
            radius = max(1, int(math.ceil(
                self.goal_search_radius / grid.resolution)))
            clearance = max(0, int(math.ceil(
                self.goal_clearance / grid.resolution)))
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    candidate = (
                        center_cell[0] + dx, center_cell[1] + dy)
                    if (not self._inside(grid, candidate) or
                            math.hypot(dx, dy) > radius or
                            int(grid.data[
                                candidate[1], candidate[0]]) != 0):
                        continue
                    safe = True
                    for oy in range(-clearance, clearance + 1):
                        for ox in range(-clearance, clearance + 1):
                            if (math.hypot(ox, oy) * grid.resolution >
                                    self.goal_clearance):
                                continue
                            probe = (
                                candidate[0] + ox, candidate[1] + oy)
                            if (not self._inside(grid, probe) or
                                    int(grid.data[
                                        probe[1], probe[0]]) != 0):
                                safe = False
                                break
                        if not safe:
                            break
                    if safe:
                        search_cells.add(candidate)
        reachable_candidates = list(search_cells & reachable_free)
        candidates = reachable_candidates or list(search_cells)
        if not candidates:
            return None
        goal_cell = min(
            candidates,
            key=lambda cell: math.hypot(
                grid.cell_to_world(cell)[0] - center[0],
                grid.cell_to_world(cell)[1] - center[1]))
        goal = grid.cell_to_world(goal_cell)
        distance = math.hypot(
            goal[0] - float(robot_pose[0]),
            goal[1] - float(robot_pose[1]))
        return FrontierCluster(
            id="frontier_{:06d}_{:03d}".format(self.update_sequence, index),
            center=(float(center[0]), float(center[1])),
            goal=(float(goal[0]), float(goal[1])),
            size=len(cells),
            unknown_area=self._unknown_area(grid, cells),
            boundary_length=len(cells) * grid.resolution,
            reachable=bool(reachable_candidates),
            distance_to_robot=distance,
            cells=list(cells),
            free_cells=sorted(search_cells),
        )

    def _merge(self, grid, clusters: List[FrontierCluster],
               robot_pose: Sequence[float]) -> List[FrontierCluster]:
        groups: List[List[FrontierCluster]] = []
        for cluster in clusters:
            selected = None
            for group in groups:
                if any(
                    math.hypot(cluster.center[0] - other.center[0],
                               cluster.center[1] - other.center[1]) <=
                    self.merge_distance and
                    cluster.reachable == other.reachable
                    for other in group
                ):
                    selected = group
                    break
            if selected is None:
                groups.append([cluster])
            else:
                selected.append(cluster)
        merged = []
        for index, group in enumerate(groups):
            if len(group) == 1:
                merged.append(group[0])
                continue
            cells = sorted({
                cell for cluster in group for cell in cluster.cells})
            item = self._build_cluster(
                grid, cells,
                {cell for cluster in group if cluster.reachable
                 for cell in cluster.free_cells},
                robot_pose, index)
            if item is not None:
                item.id = group[0].id
                item.reachable = any(cluster.reachable for cluster in group)
                merged.append(item)
        return merged

    def detect(self, grid, robot_pose: Sequence[float]
               ) -> List[FrontierCluster]:
        self.update_sequence += 1
        data = grid.data
        unknown = data < 0
        free = data == 0
        frontier_mask = np.zeros(data.shape, dtype=bool)
        frontier_mask[1:, :] |= unknown[1:, :] & free[:-1, :]
        frontier_mask[:-1, :] |= unknown[:-1, :] & free[1:, :]
        frontier_mask[:, 1:] |= unknown[:, 1:] & free[:, :-1]
        frontier_mask[:, :-1] |= unknown[:, :-1] & free[:, 1:]
        remaining = {
            (int(x), int(y))
            for y, x in np.argwhere(frontier_mask)}
        reachable_free = self._reachable_free(grid, robot_pose)
        clusters: List[FrontierCluster] = []
        while remaining:
            seed = remaining.pop()
            queue, cells = deque([seed]), [seed]
            while queue:
                cell = queue.popleft()
                for nxt in self._neighbors8(cell):
                    if nxt not in remaining:
                        continue
                    remaining.remove(nxt)
                    cells.append(nxt)
                    queue.append(nxt)
            if len(cells) < self.minimum_cluster_cells:
                continue
            cluster = self._build_cluster(
                grid, cells, reachable_free, robot_pose, len(clusters))
            if cluster is not None:
                clusters.append(cluster)
        return self._merge(grid, clusters, robot_pose)


def detect_frontier_clusters(grid, robot_pose: Sequence[float],
                             minimum_cluster_cells: int = 8,
                             merge_distance: float = 0.75,
                             unknown_area_radius: float = 2.0
                             ) -> List[FrontierCluster]:
    return FrontierClusterDetector(
        minimum_cluster_cells, merge_distance,
        unknown_area_radius).detect(grid, robot_pose)
