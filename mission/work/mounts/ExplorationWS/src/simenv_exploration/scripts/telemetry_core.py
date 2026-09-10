#!/usr/bin/env python3
"""Small ROS-free telemetry primitives used by baseline logging and tests."""

from __future__ import annotations

import json
import math
import os
import queue
import threading
import time
from typing import Iterable, Optional, Sequence


def integrate_trajectory(records: Iterable[dict]) -> float:
    points = []
    for record in records:
        try:
            point = (float(record["position_x"]), float(record["position_y"]))
        except (KeyError, TypeError, ValueError):
            continue
        if all(math.isfinite(value) for value in point):
            points.append(point)
    return sum(math.hypot(b[0] - a[0], b[1] - a[1])
               for a, b in zip(points[:-1], points[1:]))


def growth_rates(records: Sequence[dict], field: str = "observed_voxels") -> list:
    rates = []
    for first, second in zip(records[:-1], records[1:]):
        try:
            dt = float(second["elapsed_time"]) - float(first["elapsed_time"])
            gain = float(second[field]) - float(first[field])
        except (KeyError, TypeError, ValueError):
            rates.append(None)
            continue
        rates.append(gain / dt if dt > 0.0 else None)
    return rates


class AsyncJsonlWriter:
    """Bounded background JSONL writer; logging failure never raises to control."""
    def __init__(self, path: str, capacity: int = 2048, flush_every: int = 10):
        self.path = path
        self.queue = queue.Queue(maxsize=max(1, capacity))
        self.flush_every = max(1, flush_every)
        self.errors = []
        self.dropped = 0
        self.durations_ms = []
        self._stop = object()
        self._thread = threading.Thread(target=self._run, name="jsonl-writer", daemon=True)
        self._thread.start()

    def submit(self, record: dict) -> bool:
        try:
            self.queue.put_nowait(dict(record))
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def _run(self) -> None:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as stream:
                count = 0
                while True:
                    item = self.queue.get()
                    if item is self._stop:
                        break
                    started = time.perf_counter()
                    stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
                    count += 1
                    if count % self.flush_every == 0:
                        stream.flush()
                    self.durations_ms.append((time.perf_counter() - started) * 1000.0)
                stream.flush()
        except (OSError, ValueError) as error:
            self.errors.append(str(error))

    def close(self, timeout: float = 2.0) -> None:
        try:
            self.queue.put(self._stop, timeout=max(0.1, timeout))
        except queue.Full:
            self.errors.append("close queue timeout")
            return
        self._thread.join(timeout=max(0.1, timeout))

    @property
    def mean_ms(self) -> Optional[float]:
        return sum(self.durations_ms) / len(self.durations_ms) if self.durations_ms else None

    @property
    def max_ms(self) -> Optional[float]:
        return max(self.durations_ms) if self.durations_ms else None
