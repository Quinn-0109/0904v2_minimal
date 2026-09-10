"""Pure geometry helpers shared by the UFO path adapter and tests."""

import math


def distance(a, b):
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def trim_path_to_pose(points, pose):
    path = [(float(p[0]), float(p[1])) for p in points]
    if not path:
        return []
    robot = (float(pose[0]), float(pose[1]))
    if len(path) == 1:
        return [robot, path[0]] if distance(robot, path[0]) > 1e-6 else [robot]
    best = None
    for index, (start, end) in enumerate(zip(path[:-1], path[1:])):
        dx, dy = end[0] - start[0], end[1] - start[1]
        denom = dx * dx + dy * dy
        ratio = 0.0 if denom <= 1e-12 else max(0.0, min(
            1.0, ((robot[0] - start[0]) * dx +
                  (robot[1] - start[1]) * dy) / denom))
        projection = (start[0] + ratio * dx, start[1] + ratio * dy)
        candidate = (distance(robot, projection), index, ratio, projection)
        if best is None or candidate[:3] < best[:3]:
            best = candidate
    _, index, ratio, projection = best
    result = [robot]
    if distance(robot, projection) > 1e-3:
        result.append(projection)
    first = index + 1 + (1 if ratio >= 1.0 - 1e-9 else 0)
    for point in path[first:]:
        if distance(result[-1], point) > 1e-3:
            result.append(point)
    return result


def resample_path(points, spacing=0.45):
    if not points:
        return []
    spacing = max(0.05, float(spacing))
    output = [tuple(points[0])]
    carry = 0.0
    for start, end in zip(points[:-1], points[1:]):
        length = distance(start, end)
        if length <= 1e-9:
            continue
        dx = (end[0] - start[0]) / length
        dy = (end[1] - start[1]) / length
        travelled = spacing - carry if carry > 1e-9 else spacing
        while travelled <= length + 1e-9:
            point = (start[0] + travelled * dx, start[1] + travelled * dy)
            if distance(output[-1], point) > 1e-3:
                output.append(point)
            travelled += spacing
        carry = max(0.0, length - (travelled - spacing))
    if distance(output[-1], points[-1]) > spacing * 0.35:
        output.append(tuple(points[-1]))
    return output


def crop_horizon(points, horizon=3.0):
    if not points:
        return []
    result = [tuple(points[0])]
    remaining = max(0.0, float(horizon))
    for start, end in zip(points[:-1], points[1:]):
        segment = distance(start, end)
        if segment <= remaining + 1e-9:
            result.append(tuple(end))
            remaining -= segment
            continue
        if segment > 1e-9 and remaining > 1e-9:
            ratio = remaining / segment
            result.append((start[0] + ratio * (end[0] - start[0]),
                           start[1] + ratio * (end[1] - start[1])))
        break
    return result


def select_execution_waypoints(points, pose, min_distance=0.6):
    selected = []
    robot = (float(pose[0]), float(pose[1]))
    started = False
    for index, point in enumerate(points[1:], 1):
        if not started and distance(robot, point) + 1e-9 < float(min_distance):
            continue
        started = True
        next_point = points[min(index + 1, len(points) - 1)]
        yaw = math.atan2(next_point[1] - point[1], next_point[0] - point[0])
        selected.append((float(point[0]), float(point[1]), yaw))
    return selected
