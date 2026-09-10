#!/usr/bin/env python3
"""Small serialization helpers shared by hierarchical exploration ROS nodes."""

import json
import os
import tempfile

import numpy as np

from baseline_planning_core import OccupancyGrid2D
from exploration_graph import FrontierCluster


def occupancy_from_message(message):
    data = np.asarray(message.data, dtype=np.int16).reshape(
        (message.info.height, message.info.width))
    return OccupancyGrid2D(
        data=data,
        resolution=float(message.info.resolution),
        origin_x=float(message.info.origin.position.x),
        origin_y=float(message.info.origin.position.y),
    )


def cluster_to_dict(cluster, include_cells=False):
    payload = {
        "id": cluster.id,
        "center": list(cluster.center),
        "goal": list(cluster.goal),
        "size": cluster.size,
        "unknown_area": cluster.unknown_area,
        "boundary_length": cluster.boundary_length,
        "reachable": cluster.reachable,
        "distance_to_robot": cluster.distance_to_robot,
        "region_hint": cluster.region_hint,
    }
    if include_cells:
        payload["cells"] = [list(cell) for cell in cluster.cells]
        payload["free_cells"] = [list(cell) for cell in cluster.free_cells]
    return payload


def cluster_from_dict(payload):
    return FrontierCluster(
        id=str(payload["id"]),
        center=tuple(map(float, payload["center"][:2])),
        goal=tuple(map(float, payload["goal"][:2])),
        size=int(payload.get("size", 0)),
        unknown_area=float(payload.get("unknown_area", 0.0)),
        boundary_length=float(payload.get("boundary_length", 0.0)),
        reachable=bool(payload.get("reachable", False)),
        distance_to_robot=float(payload.get("distance_to_robot", 0.0)),
        cells=[tuple(map(int, item[:2]))
               for item in payload.get("cells", [])],
        free_cells=[tuple(map(int, item[:2]))
                    for item in payload.get("free_cells", [])],
        region_hint=payload.get("region_hint"),
    )


def atomic_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=".tmp_", suffix=".json", dir=os.path.dirname(path))
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

