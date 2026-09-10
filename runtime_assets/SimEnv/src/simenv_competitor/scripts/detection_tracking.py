#!/usr/bin/env python3
"""Pure helpers for timestamped pose interpolation and multi-view clustering."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple


def angle_diff(target: float, source: float) -> float:
    return (target - source + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class PoseSample:
    stamp: float
    position: Tuple[float, float, float]
    quaternion: Tuple[float, float, float, float]


def _normalize_quaternion(
    quaternion: Sequence[float],
) -> Tuple[float, float, float, float]:
    norm = math.sqrt(sum(float(value) ** 2 for value in quaternion))
    if norm < 1e-12:
        return 0.0, 0.0, 0.0, 1.0
    return tuple(float(value) / norm for value in quaternion)  # type: ignore[return-value]


def slerp_quaternion(
    first: Sequence[float], second: Sequence[float], ratio: float
) -> Tuple[float, float, float, float]:
    """Shortest-path quaternion interpolation without ROS/tf dependencies."""
    qa = _normalize_quaternion(first)
    qb = _normalize_quaternion(second)
    dot = sum(a * b for a, b in zip(qa, qb))
    if dot < 0.0:
        qb = tuple(-value for value in qb)
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    ratio = max(0.0, min(1.0, float(ratio)))
    if dot > 0.9995:
        return _normalize_quaternion(
            tuple(a + ratio * (b - a) for a, b in zip(qa, qb))
        )
    theta = math.acos(dot)
    sine = math.sin(theta)
    weight_a = math.sin((1.0 - ratio) * theta) / sine
    weight_b = math.sin(ratio * theta) / sine
    return _normalize_quaternion(
        tuple(weight_a * a + weight_b * b for a, b in zip(qa, qb))
    )


def interpolate_pose(
    samples: Sequence[PoseSample], stamp: float, max_skew: float = 0.12
) -> Optional[PoseSample]:
    """Interpolate at sensor time, rejecting stale or widely bracketed poses."""
    if not samples or not math.isfinite(stamp):
        return None
    ordered = sorted(samples, key=lambda item: item.stamp)
    if stamp <= ordered[0].stamp:
        return ordered[0] if ordered[0].stamp - stamp <= max_skew else None
    if stamp >= ordered[-1].stamp:
        return ordered[-1] if stamp - ordered[-1].stamp <= max_skew else None
    for left, right in zip(ordered[:-1], ordered[1:]):
        if left.stamp <= stamp <= right.stamp:
            if max(stamp - left.stamp, right.stamp - stamp) > max_skew:
                return None
            duration = right.stamp - left.stamp
            if duration <= 1e-9:
                return left
            ratio = (stamp - left.stamp) / duration
            position = tuple(
                left.position[index]
                + ratio * (right.position[index] - left.position[index])
                for index in range(3)
            )
            quaternion = slerp_quaternion(left.quaternion, right.quaternion, ratio)
            return PoseSample(stamp, position, quaternion)
    return None


def yaw_from_tuple(quaternion: Sequence[float]) -> float:
    x, y, z, w = _normalize_quaternion(quaternion)
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


@dataclass
class MultiViewCluster:
    position: Tuple[float, float, float]
    hits: int = 1
    view_yaws: List[float] = field(default_factory=list)
    view_positions: List[Tuple[float, float, float]] = field(default_factory=list)
    first_stamp: float = 0.0
    last_stamp: float = 0.0

    def add(
        self,
        position: Sequence[float],
        view_yaw: float,
        stamp: float,
        min_angle: float,
        viewpoint: Optional[Sequence[float]] = None,
    ) -> None:
        self.hits += 1
        weight = 1.0 / float(self.hits)
        self.position = tuple(
            self.position[index] * (1.0 - weight) + float(position[index]) * weight
            for index in range(3)
        )
        if not any(abs(angle_diff(view_yaw, prior)) >= min_angle for prior in self.view_yaws):
            # Keep one representative for a redundant view direction.
            if not self.view_yaws:
                self.view_yaws.append(view_yaw)
        else:
            self.view_yaws.append(view_yaw)
        if viewpoint is not None:
            candidate = tuple(float(value) for value in viewpoint)
            if not self.view_positions or all(
                MultiViewTracker.distance(candidate, prior) >= 0.05
                for prior in self.view_positions
            ):
                self.view_positions.append(candidate)
        self.last_stamp = max(self.last_stamp, stamp)

    @property
    def view_count(self) -> int:
        return len(self.view_yaws)

    @property
    def viewpoint_baseline(self) -> float:
        if len(self.view_positions) < 2:
            return 0.0
        return max(
            MultiViewTracker.distance(first, second)
            for index, first in enumerate(self.view_positions)
            for second in self.view_positions[index + 1 :]
        )


class MultiViewTracker:
    def __init__(self, radius: float = 0.6, min_view_angle: float = math.radians(12.0)):
        self.radius = max(0.05, float(radius))
        self.min_view_angle = max(0.01, float(min_view_angle))
        self.clusters: List[MultiViewCluster] = []

    @staticmethod
    def distance(first: Sequence[float], second: Sequence[float]) -> float:
        return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(first, second)))

    def add(
        self,
        position: Sequence[float],
        view_yaw: float,
        stamp: float,
        viewpoint: Optional[Sequence[float]] = None,
    ) -> MultiViewCluster:
        best = None
        best_distance = None
        for cluster in self.clusters:
            distance = self.distance(position, cluster.position)
            if distance <= self.radius and (best_distance is None or distance < best_distance):
                best = cluster
                best_distance = distance
        if best is None:
            best = MultiViewCluster(
                position=tuple(float(value) for value in position),
                hits=1,
                view_yaws=[float(view_yaw)],
                view_positions=(
                    [tuple(float(value) for value in viewpoint)]
                    if viewpoint is not None
                    else []
                ),
                first_stamp=float(stamp),
                last_stamp=float(stamp),
            )
            self.clusters.append(best)
        else:
            best.add(
                position,
                float(view_yaw),
                float(stamp),
                self.min_view_angle,
                viewpoint=viewpoint,
            )
        return best

    def confirmed(
        self,
        minimum_hits: int,
        minimum_views: int,
        minimum_viewpoint_baseline: float = 0.0,
    ) -> List[MultiViewCluster]:
        return [
            cluster
            for cluster in self.clusters
            if cluster.hits >= minimum_hits
            and cluster.view_count >= minimum_views
            and cluster.viewpoint_baseline >= minimum_viewpoint_baseline
        ]
