#!/usr/bin/env python3
"""Direct oracle waypoint tracker for A1 via /cmd_vel.

Bypasses CMU localPlanner (which gets stuck on RealSense door-frame clouds).
Uses /state_estimation from the RealSense/GT bridge. NOT competition-compliant.
"""

import json
import heapq
import math
import os

import rospy
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def roll_pitch_from_quaternion(q):
    sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
    cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (q.w * q.y - q.z * q.x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    return roll, pitch


def quaternion_from_yaw(yaw):
    half = 0.5 * yaw
    return (0.0, 0.0, math.sin(half), math.cos(half))


def angle_diff(target, source):
    return (target - source + math.pi) % (2.0 * math.pi) - math.pi


class OracleWaypointTracker:
    def __init__(self):
        root = rospy.get_param(
            "~workspace_root",
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")),
        )
        layout_path = rospy.get_param(
            "~layout_file",
            os.path.join(root, "generated_building", "layout_metadata.json"),
        )
        self._goal_tolerance = float(rospy.get_param("~goal_tolerance", 0.55))
        self._max_speed = float(rospy.get_param("~max_speed", 0.45))
        self._max_yaw = float(rospy.get_param("~max_yaw_rate", 0.55))
        self._slow_yaw = float(rospy.get_param("~heading_align_rad", 0.30))
        self._start_delay = float(rospy.get_param("~start_delay", 4.0))
        self._scan_target = math.radians(
            float(rospy.get_param("~scan_angle_deg", 120.0))
        )
        self._coverage_profile = rospy.get_param("~coverage_profile", "room_sweep")
        self._floor_index = int(rospy.get_param("~floor_index", 0))
        self._waypoints = self._load_waypoints(
            layout_path,
            coverage_profile=self._coverage_profile,
            floor_index=self._floor_index,
        )
        self._index = 0
        self._pose = None
        self._finished = False
        self._scanning = False
        self._scan_progress = 0.0
        self._last_scan_yaw = None
        self._recovering = False
        self._recover_step = 0
        self._recover_step_time = rospy.Time(0)
        self._last_recover = rospy.Time(0)
        self._node_start = rospy.Time.now()
        self._finish_pub = rospy.Publisher(
            "/sensor_coverage_planner/exploration_finish",
            Bool,
            queue_size=1,
            latch=True,
        )
        self._cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self._joy_pub = rospy.Publisher("/joy", Joy, queue_size=1)
        rospy.wait_for_service("/gazebo/set_model_state", timeout=60.0)
        self._set_model_state = rospy.ServiceProxy(
            "/gazebo/set_model_state", SetModelState
        )
        rospy.Subscriber("/state_estimation", Odometry, self._on_odom, queue_size=5)
        rospy.Timer(rospy.Duration(0.05), self._on_timer)
        rospy.Timer(rospy.Duration(1.0), self._open_doors, oneshot=True)
        rospy.loginfo(
            "Oracle tracker loaded %d %s-coverage waypoints from %s (floor=%d, max_speed=%.2f)",
            len(self._waypoints),
            self._coverage_profile,
            layout_path,
            self._floor_index,
            self._max_speed,
        )
        self._log_waypoint()

    def _open_doors(self, _event=None):
        try:
            from building_generator_interfaces.srv import SetDoorState

            rospy.wait_for_service("/set_door_state", timeout=10.0)
            client = rospy.ServiceProxy("/set_door_state", SetDoorState)
            door_ids = ["elevator_floor_{}".format(self._floor_index)]
            if self._floor_index == 0:
                door_ids.insert(0, "main_entrance")
            for door_id in door_ids:
                try:
                    response = client(door_id=door_id, open=True)
                    rospy.loginfo("Door %s -> %s", door_id, response.state)
                except rospy.ServiceException as error:
                    rospy.logwarn("Failed to open %s: %s", door_id, error)
        except Exception as error:  # noqa: BLE001 - best-effort oracle helper
            rospy.logwarn("Door open helper unavailable: %s", error)

    @staticmethod
    def _load_waypoints(layout_path, coverage_profile="balanced", floor_index=0):
        with open(layout_path, encoding="utf-8") as stream:
            layout = json.load(stream)

        floor = next(
            floor
            for floor in layout.get("floors", [])
            if int(floor.get("floor_index", 0)) == int(floor_index)
        )
        rooms = sorted(
            floor.get("rooms", []),
            key=lambda item: (
                item.get("bounds", {}).get("y_min", 0.0),
                item.get("side", ""),
            ),
        )

        footprint = layout.get("footprint", {})
        x_limit = float(footprint.get("width", 20.0)) / 2.0 - 0.35
        y_limit = float(footprint.get("length", 36.0)) - 0.35
        resolution = 0.30
        inflation = 0.48
        wall_half_width = 0.18
        doors = [
            (float(room["door_pose"][0]), float(room["door_pose"][1]))
            for room in rooms
            if room.get("door_pose")
        ]
        furniture = []
        for room in rooms:
            for item in room.get("furniture", []):
                px, py = float(item["pose"][0]), float(item["pose"][1])
                sx, sy = float(item["size"][0]), float(item["size"][1])
                furniture.append(
                    (
                        px - sx / 2.0 - inflation,
                        py - sy / 2.0 - inflation,
                        px + sx / 2.0 + inflation,
                        py + sy / 2.0 + inflation,
                    )
                )

        corridor = floor["corridor_bounds"]
        lobby = floor["lobby_bounds"]
        row_boundary = min(
            room["bounds"]["y_max"]
            for room in rooms
            if room["bounds"]["y_max"] > lobby["y_max"] + 2.0
        )

        def blocked(x, y):
            if x < -x_limit or x > x_limit or y < 0.35 or y > y_limit:
                return True
            for x0, y0, x1, y1 in furniture:
                if x0 <= x <= x1 and y0 <= y <= y1:
                    return True
            # Lobby-to-room boundary; only the central corridor opening is free.
            if (
                abs(y - float(lobby["y_max"])) <= wall_half_width
                and abs(x) > float(corridor["x_max"]) - 0.05
            ):
                return True
            # Horizontal wall between the two room rows.
            if (
                abs(y - row_boundary) <= wall_half_width
                and abs(x) > float(corridor["x_max"]) - 0.05
            ):
                return True
            # Corridor side walls, except at room doors.
            for wall_x in (float(corridor["x_min"]), float(corridor["x_max"])):
                if abs(x - wall_x) <= wall_half_width and y >= corridor["y_min"]:
                    if not any(
                        abs(door_x - wall_x) < 0.1 and abs(y - door_y) <= 0.75
                        for door_x, door_y in doors
                    ):
                        return True
            return False

        def nearest_free(point):
            x, y = point
            if not blocked(x, y):
                return x, y
            for radius_step in range(1, 15):
                radius = radius_step * resolution
                for angle_step in range(24):
                    angle = 2.0 * math.pi * angle_step / 24.0
                    candidate = (
                        x + radius * math.cos(angle),
                        y + radius * math.sin(angle),
                    )
                    if not blocked(*candidate):
                        return candidate
            raise RuntimeError("No collision-free point near {}".format(point))

        def grid(point):
            return (
                int(round(point[0] / resolution)),
                int(round(point[1] / resolution)),
            )

        def world(cell):
            return cell[0] * resolution, cell[1] * resolution

        def free_cell(point):
            center = grid(nearest_free(point))
            if not blocked(*world(center)):
                return center
            for radius in range(1, 12):
                for dx in range(-radius, radius + 1):
                    for dy in (-radius, radius):
                        cell = center[0] + dx, center[1] + dy
                        if not blocked(*world(cell)):
                            return cell
                for dy in range(-radius + 1, radius):
                    for dx in (-radius, radius):
                        cell = center[0] + dx, center[1] + dy
                        if not blocked(*world(cell)):
                            return cell
            raise RuntimeError("No free grid cell near {}".format(point))

        neighbors = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

        def astar(start, goal):
            start_cell, goal_cell = free_cell(start), free_cell(goal)
            frontier = [(0.0, start_cell)]
            came_from = {start_cell: None}
            cost = {start_cell: 0.0}
            while frontier:
                _, current = heapq.heappop(frontier)
                if current == goal_cell:
                    break
                for dx, dy in neighbors:
                    nxt = current[0] + dx, current[1] + dy
                    wx, wy = world(nxt)
                    if blocked(wx, wy):
                        continue
                    step_cost = math.sqrt(2.0) if dx and dy else 1.0
                    new_cost = cost[current] + step_cost
                    if nxt not in cost or new_cost < cost[nxt]:
                        cost[nxt] = new_cost
                        heuristic = math.hypot(
                            goal_cell[0] - nxt[0], goal_cell[1] - nxt[1]
                        )
                        heapq.heappush(frontier, (new_cost + heuristic, nxt))
                        came_from[nxt] = current
            if goal_cell not in came_from:
                raise RuntimeError("A* failed from {} to {}".format(start, goal))
            cells = []
            current = goal_cell
            while current is not None:
                cells.append(current)
                current = came_from[current]
            cells.reverse()
            return [world(cell) for cell in cells]

        def line_free(start, end):
            distance = math.hypot(end[0] - start[0], end[1] - start[1])
            steps = max(1, int(distance / 0.10))
            for step in range(1, steps + 1):
                ratio = step / steps
                if blocked(
                    start[0] + (end[0] - start[0]) * ratio,
                    start[1] + (end[1] - start[1]) * ratio,
                ):
                    return False
            return True

        def simplify(points):
            if len(points) <= 2:
                return points
            result = [points[0]]
            index = 0
            while index < len(points) - 1:
                candidate = len(points) - 1
                while candidate > index + 1:
                    if line_free(points[index], points[candidate]):
                        break
                    candidate -= 1
                result.append(points[candidate])
                index = candidate
            return result

        # Fast coverage deliberately visits every room before any room gets a
        # second observation point.  It is useful when the mission time is
        # tightly limited.
        first_pass = [((0.0, 4.0), False), ((0.0, 7.0), False)]
        room_sweep = [((0.0, 4.0), False), ((0.0, 7.0), False)]
        refinement_pass = []
        for room in rooms:
            bounds = room["bounds"]
            cx = 0.5 * (bounds["x_min"] + bounds["x_max"])
            cy = 0.5 * (bounds["y_min"] + bounds["y_max"])
            door = room.get("door_pose")
            corridor_y = float(door[1]) if door else cy
            if door:
                side_sign = -1.0 if door[0] < 0 else 1.0
                door_entry = (
                    (float(door[0]) + side_sign * 0.7, float(door[1])),
                    False,
                )
            else:
                door_entry = ((cx, corridor_y), False)
            observation_y = float(bounds["y_min"]) + 2.2
            center = (nearest_free((cx, cy)), True)

            first_pass.append(((0.0, corridor_y), False))
            first_pass.append(door_entry)
            first_pass.append(center)
            first_pass.append(((0.0, corridor_y), False))

            # A stable zig-zag inside one room before crossing back through
            # its door.  This avoids the failed cross-room refinement
            # pattern, while seeing behind the near furniture row.
            room_sweep.extend(
                [
                    ((0.0, corridor_y), False),
                    door_entry,
                    (nearest_free((cx, observation_y)), False),
                    center,
                    ((0.0, corridor_y), False),
                ]
            )
            if coverage_profile == "thorough":
                refinement_pass.extend(
                    [
                        ((0.0, corridor_y), False),
                        (nearest_free((cx, observation_y)), True),
                        ((0.0, corridor_y), False),
                    ]
                )

        targets = room_sweep if coverage_profile == "room_sweep" else first_pass
        if coverage_profile == "thorough":
            targets += refinement_pass
        targets.extend(
            [((0.0, 7.0), False), ((0.0, 4.0), False), ((0.0, 2.0), False)]
        )

        waypoints = []
        current = (0.0, 2.0)
        for target, scan_direction in targets:
            path = simplify(astar(current, target))
            for point in path[1:-1]:
                waypoints.append(
                    {
                        "x": point[0],
                        "y": point[1],
                        "scan": False,
                        "scan_direction": 0.0,
                    }
                )
            target = nearest_free(target)
            waypoints.append(
                {
                    "x": target[0],
                    "y": target[1],
                    "scan": bool(scan_direction),
                    "scan_direction": float(scan_direction),
                }
            )
            current = target
        return waypoints

    def _log_waypoint(self):
        if self._index >= len(self._waypoints):
            return
        waypoint = self._waypoints[self._index]
        x, y = waypoint["x"], waypoint["y"]
        rospy.loginfo(
            "Oracle waypoint %d/%d -> (%.2f, %.2f)%s",
            self._index + 1,
            len(self._waypoints),
            x,
            y,
            " [{:.0f}-degree scan]".format(math.degrees(self._scan_target))
            if waypoint["scan"]
            else "",
        )

    def _on_odom(self, message):
        self._pose = message

    def _publish_joy(self, button_index):
        message = Joy()
        message.header.stamp = rospy.Time.now()
        message.axes = [0.0] * 8
        message.buttons = [0] * 12
        if 0 <= button_index < len(message.buttons):
            message.buttons[button_index] = 1
        self._joy_pub.publish(message)

    def _upright_in_place(self):
        if self._pose is None:
            return False
        position = self._pose.pose.pose.position
        yaw = yaw_from_quaternion(self._pose.pose.pose.orientation)
        state = ModelState()
        state.model_name = "a1_gazebo"
        state.reference_frame = "world"
        state.pose.position.x = position.x
        state.pose.position.y = position.y
        state.pose.position.z = 0.55
        quat = quaternion_from_yaw(yaw)
        state.pose.orientation.x = quat[0]
        state.pose.orientation.y = quat[1]
        state.pose.orientation.z = quat[2]
        state.pose.orientation.w = quat[3]
        try:
            response = self._set_model_state(state)
            return bool(response.success)
        except rospy.ServiceException as error:
            rospy.logwarn("set_model_state failed: %s", error)
            return False

    def _recover_from_fall(self):
        now = rospy.Time.now()
        if not self._recovering:
            if (now - self._last_recover).to_sec() < 6.0:
                return
            self._recovering = True
            self._recover_step = 0
            self._recover_step_time = now
            self._last_recover = now
            rospy.logwarn("Fall detected; starting upright/stand/RL recovery.")
            self._cmd_pub.publish(Twist())
            self._upright_in_place()
            return

        elapsed = (now - self._recover_step_time).to_sec()
        if self._recover_step == 0 and elapsed >= 0.5:
            self._publish_joy(1)  # fixed stand
            self._recover_step = 1
            self._recover_step_time = now
        elif self._recover_step == 1 and elapsed >= 3.0:
            self._publish_joy(3)  # RL /cmd_vel
            self._recover_step = 2
            self._recover_step_time = now
        elif self._recover_step == 2 and elapsed >= 1.0:
            self._recovering = False
            self._recover_step = 0
            rospy.loginfo("Recovery complete; resuming waypoint tracking.")

    def _is_fallen(self):
        if self._pose is None:
            return False
        position = self._pose.pose.pose.position
        orientation = self._pose.pose.pose.orientation
        roll, pitch = roll_pitch_from_quaternion(orientation)
        return position.z < 0.22 or abs(roll) > 0.85 or abs(pitch) > 0.85

    def _advance_waypoint(self):
        self._scanning = False
        self._scan_progress = 0.0
        self._last_scan_yaw = None
        self._index += 1
        if self._index >= len(self._waypoints):
            self._finished = True
            rospy.loginfo("Oracle room tour finished.")
            self._finish_pub.publish(Bool(data=True))
            return False
        self._log_waypoint()
        return True

    def _run_scan(self, yaw, command):
        if self._last_scan_yaw is None:
            self._last_scan_yaw = yaw
        else:
            delta = abs(angle_diff(yaw, self._last_scan_yaw))
            # Reject an odometry discontinuity caused by fall recovery.
            if delta < 0.5:
                self._scan_progress += delta
            self._last_scan_yaw = yaw
        if self._scan_progress >= self._scan_target:
            rospy.loginfo(
                "Completed %.0f-degree room scan.", math.degrees(self._scan_target)
            )
            self._advance_waypoint()
            return
        direction = self._waypoints[self._index].get("scan_direction", 1.0)
        command.angular.z = self._max_yaw * direction
        command.linear.x = 0.0

    def _on_timer(self, _event):
        command = Twist()
        if self._finished or self._pose is None:
            self._cmd_pub.publish(command)
            return

        if self._recovering or self._is_fallen():
            self._cmd_pub.publish(command)
            self._recover_from_fall()
            return

        if (rospy.Time.now() - self._node_start).to_sec() < self._start_delay:
            self._cmd_pub.publish(command)
            return

        if self._index >= len(self._waypoints):
            self._finished = True
            rospy.loginfo("Oracle room tour finished.")
            self._finish_pub.publish(Bool(data=True))
            self._cmd_pub.publish(command)
            return

        position = self._pose.pose.pose.position
        yaw = yaw_from_quaternion(self._pose.pose.pose.orientation)
        waypoint = self._waypoints[self._index]
        target_x, target_y = waypoint["x"], waypoint["y"]
        dx = target_x - position.x
        dy = target_y - position.y
        distance = math.hypot(dx, dy)

        if self._scanning:
            self._run_scan(yaw, command)
            self._cmd_pub.publish(command)
            return

        if distance <= self._goal_tolerance:
            if waypoint["scan"]:
                self._scanning = True
                self._scan_progress = 0.0
                self._last_scan_yaw = yaw
                rospy.loginfo(
                    "Starting %.0f-degree room scan.",
                    math.degrees(self._scan_target),
                )
            else:
                self._advance_waypoint()
            self._cmd_pub.publish(command)
            return

        heading = math.atan2(dy, dx)
        err = angle_diff(heading, yaw)
        command.angular.z = max(-self._max_yaw, min(self._max_yaw, 1.8 * err))
        if abs(err) < self._slow_yaw:
            scale = max(0.2, 1.0 - abs(err) / max(self._slow_yaw, 1e-3))
            speed = min(self._max_speed, 0.25 + 0.2 * distance) * scale
            command.linear.x = speed
        else:
            # Turn in place first; avoids RL falls from simultaneous yaw+drive.
            command.linear.x = 0.0
        self._cmd_pub.publish(command)


if __name__ == "__main__":
    rospy.init_node("oracle_room_tour")
    OracleWaypointTracker()
    rospy.spin()
