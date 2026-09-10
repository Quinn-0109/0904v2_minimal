#!/usr/bin/env python3
"""Official FAR Planner adapter and execution coordinator.

TARE exploration waypoints are submitted to FAR.  FAR's real visibility-graph
LINE_STRIP is converted to nav_msgs/Path, checked/refined by SCAN-lite, and
executed one look-ahead waypoint at a time by the existing Goal Executor.
A* is invoked only if FAR does not return a usable path before the timeout.
"""

import copy
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time

import rospy
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from std_msgs.msg import Bool, String
from visualization_msgs.msg import Marker

# catkin's devel-space Python relay does not automatically add the source
# script directory.  Keep the existing pure planning/safety modules importable
# without requiring a user-managed PYTHONPATH.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from baseline_planning_core import execution_waypoints
from hierarchical_ros_utils import occupancy_from_message
from planner_interface import AStarPlanner
from scan_lite_path_refiner import FootprintCheck, PathRefiner, RefinerConfig
from simenv_competitor.srv import CheckTwinCylinder, CheckTwinCylinderRequest


def _distance(a, b):
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def far_path_makes_goal_progress(points, goal, exact_match_distance,
                                 adjusted_goal_max_distance,
                                 minimum_progress):
    """Accept an exact FAR route or a bounded, useful adjusted-goal route.

    Official FAR may move an unreachable/unknown requested endpoint by its
    configured goal-adjust radius and return a safe graph path to that local
    endpoint.  The adapter executes only a short SCAN-lite-validated prefix,
    so rejecting every adjusted endpoint prevents the graph from growing
    toward the TARE goal.  A partial path is usable only when its endpoint is
    within FAR's adjustment bound and measurably closer to the unchanged TARE
    goal than the live path start.
    """
    if not points:
        return False
    endpoint_distance = _distance(points[-1], goal)
    if endpoint_distance <= float(exact_match_distance):
        return True
    if endpoint_distance > float(adjusted_goal_max_distance):
        return False
    return (_distance(points[0], goal) - endpoint_distance >=
            float(minimum_progress))


def trim_path_to_pose(points, pose, duplicate_tolerance=0.03):
    """Drop the already-traversed prefix of a planner polyline.

    FAR may keep publishing a path whose first vertex is the pose at which the
    goal was originally submitted.  Reusing that complete polyline after each
    local waypoint makes the executor select the same look-ahead point forever.
    Project the live robot pose onto the nearest segment and retain only the
    forward suffix.  The live pose is prepended so distance accumulation starts
    at the robot, not at a stale graph vertex; the already-occupied first point
    is removed before SCAN-lite validates the executable suffix.
    """
    path = [(float(point[0]), float(point[1])) for point in points]
    if pose is None or not path:
        return path
    robot = (float(pose[0]), float(pose[1]))
    if len(path) == 1:
        return ([robot] if _distance(robot, path[0]) <= duplicate_tolerance
                else [robot, path[0]])

    best = None
    for index, (start, end) in enumerate(zip(path[:-1], path[1:])):
        dx, dy = end[0] - start[0], end[1] - start[1]
        length_squared = dx * dx + dy * dy
        if length_squared <= 1e-12:
            fraction = 0.0
        else:
            fraction = max(0.0, min(1.0, (
                (robot[0] - start[0]) * dx +
                (robot[1] - start[1]) * dy) / length_squared))
        projection = (start[0] + fraction * dx, start[1] + fraction * dy)
        candidate = (_distance(robot, projection), index, fraction, projection)
        if best is None or candidate[:3] < best[:3]:
            best = candidate

    _, segment_index, fraction, projection = best
    suffix = [robot]
    if _distance(robot, projection) > duplicate_tolerance:
        suffix.append(projection)
    next_index = segment_index + 1
    # A projection exactly on the segment's start must retain its end; an
    # exact projection on its end naturally starts with the following vertex.
    if fraction >= 1.0 - 1e-9:
        next_index += 1
    for point in path[next_index:]:
        if _distance(point, suffix[-1]) > duplicate_tolerance:
            suffix.append(point)
    return suffix


class FarInterface:
    def __init__(self):
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._pose = None
        self._home_pose = None
        self._pose_frame = rospy.get_param("~world_frame", "map")
        self._grid = None
        self._latest_tare_goal = None
        self._tare_goal_anchor = None
        self._goal_generation = 0
        self._far_marker = None
        self._far_marker_time = 0.0
        self._execution_results = []
        self._tare_goal_history = []
        self._execution_goal_history = []
        self._locomotion_ready = False
        self._finished = False
        self._tare_finished = False
        self._started = None

        self._output_dir = os.path.abspath(rospy.get_param(
            "~output_dir", "/tmp/simenv/tare_far"))
        os.makedirs(os.path.join(self._output_dir, "logs"), exist_ok=True)
        self._maximum_duration = float(rospy.get_param(
            "~maximum_duration", 300.0))
        self._started = time.monotonic()
        self._far_timeout = float(rospy.get_param(
            "~far_plan_timeout", 5.0))
        self._goal_timeout = float(rospy.get_param(
            "~goal_timeout", 90.0))
        self._lookahead_distance = float(rospy.get_param(
            "~execution_lookahead_distance", 1.2))
        self._minimum_execution_progress = float(rospy.get_param(
            "~minimum_execution_progress", 0.25))
        self._scan_horizon_distance = float(rospy.get_param(
            "~scan_lite_horizon_distance", 2.0))
        self._waypoint_spacing = float(rospy.get_param(
            "~waypoint_spacing", 0.45))
        self._same_tare_goal_distance = float(rospy.get_param(
            "~same_tare_goal_distance", 0.35))
        # Official TARE continuously republishes a rolling look-ahead goal.
        # A new publication must update the next planning cycle, but it must
        # not cancel a FAR/SCAN-validated local segment that is already being
        # planned or executed.  Safety rejection and execution timeouts still
        # cancel independently.  Keep the old preemptive behaviour available
        # only as an explicit diagnostic option.
        self._preempt_on_tare_goal_change = bool(rospy.get_param(
            "~preempt_on_tare_goal_change", False))
        self._minimum_path_points = int(rospy.get_param(
            "~minimum_far_path_points", 2))
        self._far_goal_match_distance = float(rospy.get_param(
            "~far_goal_match_distance", 1.5))
        self._far_adjusted_goal_max_distance = float(rospy.get_param(
            "~far_adjusted_goal_max_distance", 2.25))
        self._auto_visualization = bool(rospy.get_param(
            "~auto_generate_visualization", True))
        default_visualizer = os.path.abspath(os.path.join(
            SCRIPT_DIR, "..", "..", "..", "scripts",
            "visualize_baseline_results.py"))
        self._visualization_script = os.path.abspath(rospy.get_param(
            "~visualization_script", default_visualizer))
        self._visualization_launcher = os.path.join(
            os.path.dirname(self._visualization_script),
            "run_visualization_after_shutdown.py")
        self._truth_layout = os.path.abspath(rospy.get_param(
            "~offline_truth_layout_metadata", ""))
        if self._truth_layout and os.path.isfile(self._truth_layout):
            try:
                shutil.copy2(
                    self._truth_layout,
                    os.path.join(self._output_dir, "layout_metadata.json"))
            except OSError as error:
                rospy.logwarn(
                    "Could not snapshot offline layout for visualization: %s",
                    error)

        self._astar = AStarPlanner(
            clearance=float(rospy.get_param("~astar_clearance", 0.30)),
            reached_tolerance=float(rospy.get_param(
                "~reached_tolerance", 0.15)),
            maximum_expansions=int(rospy.get_param(
                "~astar_maximum_expansions", 100000)),
            allow_blocked_start=bool(rospy.get_param(
                "~astar_allow_blocked_start", True)))
        self._scan_config = RefinerConfig(
            unknown_policy=rospy.get_param(
                "~scan_lite_unknown_policy", "penalize"),
            body_safety_margin=float(rospy.get_param(
                "~scan_lite_body_safety_margin", 0.05)),
            refinement_search_radius=float(rospy.get_param(
                "~scan_lite_search_radius", 0.60)),
            refinement_search_step=float(rospy.get_param(
                "~scan_lite_search_step", 0.15)))
        self._scan_service = rospy.ServiceProxy(
            rospy.get_param(
                "~scan_service",
                "/tare_far_voxel_mapper/check_twin_cylinder"),
            CheckTwinCylinder)

        self._far_goal_pub = rospy.Publisher(
            "/far/goal_point", PointStamped, queue_size=1)
        self._path_pub = rospy.Publisher(
            "/simenv/far_path", Path, queue_size=2, latch=True)
        self._goal_pub = rospy.Publisher(
            "/exploration_goal", PoseStamped, queue_size=1)
        self._status_pub = rospy.Publisher(
            "/simenv/tare_far_status", String, queue_size=5, latch=True)
        self._cancel_pub = rospy.Publisher(
            "/simenv/cancel_exploration_goal", String, queue_size=2)
        self._finalize_pub = rospy.Publisher(
            "/simenv/finalize_voxel_map", Bool, queue_size=1, latch=True)
        self._complete_pub = rospy.Publisher(
            "/simenv/mission_complete", Bool, queue_size=1, latch=True)

        rospy.Subscriber(
            rospy.get_param("~tare_goal_topic", "/tare/way_point"),
            PointStamped, self._on_tare_goal, queue_size=5)
        rospy.Subscriber(
            rospy.get_param("~far_path_marker_topic", "/far/viz_path_topic"),
            Marker, self._on_far_marker, queue_size=5)
        rospy.Subscriber(
            rospy.get_param("~odom_topic", "/Odometry"),
            Odometry, self._on_odom, queue_size=30)
        rospy.Subscriber(
            rospy.get_param(
                "~map_topic", "/simenv/voxel_floor_projection"),
            OccupancyGrid, self._on_map, queue_size=1)
        rospy.Subscriber(
            "/simenv/goal_execution_result", String,
            self._on_execution_result, queue_size=20)
        rospy.Subscriber(
            "/locomotion_ready", Bool, self._on_locomotion, queue_size=2)
        rospy.Subscriber(
            rospy.get_param(
                "~tare_finish_topic",
                "/sensor_coverage_planner/exploration_finish"),
            Bool, self._on_tare_finish, queue_size=2)
        rospy.on_shutdown(self._write_summary)
        # Independent timer keeps the 300 s budget hard even if an external
        # FAR/SCAN service call is slow or blocked.
        self._duration_timer = rospy.Timer(
            rospy.Duration(0.25), self._on_duration_timer)
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()
        rospy.loginfo(
            "FAR interface ready: official FAR timeout=%.1fs A* fallback "
            "only, rolling-goal preemption=%s",
            self._far_timeout, self._preempt_on_tare_goal_change)

    def _status(self, state, **extra):
        data = {"state": state, "stamp": rospy.Time.now().to_sec()}
        data.update(extra)
        self._status_pub.publish(String(data=json.dumps(data)))

    def _on_odom(self, message):
        position = message.pose.pose.position
        with self._condition:
            self._pose = (position.x, position.y, position.z)
            if self._home_pose is None:
                self._home_pose = self._pose
            self._pose_frame = message.header.frame_id or self._pose_frame
            self._condition.notify_all()

    def _on_map(self, message):
        try:
            grid = occupancy_from_message(message)
        except (IndexError, TypeError, ValueError):
            return
        with self._condition:
            self._grid = grid
            self._condition.notify_all()

    def _on_locomotion(self, message):
        with self._condition:
            self._locomotion_ready = bool(message.data)
            self._condition.notify_all()

    def _on_tare_goal(self, message):
        goal = (float(message.point.x), float(message.point.y),
                float(message.point.z))
        with self._condition:
            if (self._tare_goal_anchor is not None and
                    _distance(goal, self._tare_goal_anchor) <
                    self._same_tare_goal_distance):
                # TARE publishes a rolling look-ahead point.  Keep the newest
                # point for the next planning cycle without cancelling the
                # waypoint already being executed.
                self._latest_tare_goal = goal
                return
            self._latest_tare_goal = goal
            self._tare_goal_anchor = goal
            self._goal_generation += 1
            self._tare_goal_history.append({
                "generation": self._goal_generation,
                "elapsed_sec": time.monotonic() - self._started,
                "ros_time": message.header.stamp.to_sec(),
                "frame_id": message.header.frame_id or self._pose_frame,
                "goal": list(goal),
                "robot_pose": (
                    list(self._pose) if self._pose is not None else None),
            })
            self._condition.notify_all()

    def _on_far_marker(self, message):
        if message.type != Marker.LINE_STRIP:
            return
        with self._condition:
            self._far_marker = copy.deepcopy(message)
            self._far_marker_time = time.monotonic()
            self._condition.notify_all()

    def _on_execution_result(self, message):
        try:
            result = json.loads(message.data)
        except (TypeError, ValueError):
            return
        with self._condition:
            self._execution_results.append(result)
            self._condition.notify_all()

    def _on_tare_finish(self, message):
        if message.data:
            with self._condition:
                self._tare_finished = True
                self._condition.notify_all()
            self._status("tare_returning_home")

    def _on_duration_timer(self, _event):
        if (not self._finished and
                time.monotonic() - self._started >= self._maximum_duration):
            self._finish("time_limit")

    def _marker_points(self, marker):
        with self._lock:
            pose = self._pose
        points = [(float(p.x), float(p.y)) for p in marker.points]
        if pose is not None and len(points) >= 2:
            robot = (pose[0], pose[1])
            if _distance(points[-1], robot) < _distance(points[0], robot):
                points.reverse()
        output = []
        for point in points:
            if not output or _distance(point, output[-1]) > 0.03:
                output.append(point)
        return trim_path_to_pose(output, pose)

    def _path_message(self, points, frame):
        message = Path()
        message.header.stamp = rospy.Time.now()
        message.header.frame_id = frame
        for index, point in enumerate(points):
            pose = PoseStamped()
            pose.header = copy.deepcopy(message.header)
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            if index + 1 < len(points):
                yaw = math.atan2(
                    points[index + 1][1] - point[1],
                    points[index + 1][0] - point[0])
                pose.pose.orientation.z = math.sin(yaw * 0.5)
                pose.pose.orientation.w = math.cos(yaw * 0.5)
            else:
                pose.pose.orientation.w = 1.0
            message.poses.append(pose)
        return message

    def _request_far(self, goal, generation, goal_match_distance=None):
        request = PointStamped()
        request.header.stamp = rospy.Time.now()
        request.header.frame_id = self._pose_frame
        request.point.x, request.point.y, request.point.z = goal
        requested_at = time.monotonic()
        self._far_goal_pub.publish(request)
        self._status("waiting_for_far", generation=generation,
                     goal=list(goal))
        deadline = requested_at + self._far_timeout
        with self._condition:
            while not rospy.is_shutdown() and time.monotonic() < deadline:
                if (self._preempt_on_tare_goal_change and
                        generation != self._goal_generation):
                    return None, "tare_goal_superseded"
                marker = copy.deepcopy(self._far_marker)
                marker_time = self._far_marker_time
                if marker is not None and marker_time >= requested_at:
                    points = self._marker_points(marker)
                    match_distance = (
                        self._far_goal_match_distance
                        if goal_match_distance is None else
                        float(goal_match_distance))
                    goal_matches = far_path_makes_goal_progress(
                        points, goal, match_distance,
                        self._far_adjusted_goal_max_distance,
                        self._minimum_execution_progress)
                    if (len(points) >= self._minimum_path_points and
                            goal_matches):
                        self._path_pub.publish(self._path_message(
                            points,
                            marker.header.frame_id or self._pose_frame))
                        return points, "far"
                self._condition.wait(timeout=0.10)
        return None, "far_timeout"

    def _fallback_astar(self, goal):
        with self._lock:
            pose, grid = self._pose, self._grid
        if pose is None or grid is None:
            return None, "astar_inputs_unavailable"
        result = self._astar.plan(pose, goal, grid)
        if not result.success or not result.path:
            return None, "astar_" + result.reason
        self._path_pub.publish(self._path_message(
            result.path, self._pose_frame))
        return result.path, "astar_fallback"

    def _scan_check(self, x, y, yaw):
        request = CheckTwinCylinderRequest()
        request.pose_x, request.pose_y = x, y
        with self._lock:
            request.pose_z = self._pose[2] if self._pose else 0.0
        request.yaw = yaw
        request.front_offset = self._scan_config.body_front_offset
        request.rear_offset = self._scan_config.body_rear_offset
        request.radius = (
            self._scan_config.body_collision_radius +
            self._scan_config.body_safety_margin)
        request.min_height = (
            self._scan_config.body_min_height -
            self._scan_config.body_safety_margin)
        request.max_height = (
            self._scan_config.body_max_height +
            self._scan_config.body_safety_margin)
        request.clearance_search_radius = request.radius + 0.50
        try:
            response = self._scan_service(request)
        except rospy.ServiceException:
            return FootprintCheck(
                map_available=False, status="map_unavailable")
        clearance = float(response.minimum_obstacle_clearance)
        return FootprintCheck(
            map_available=bool(response.map_available),
            occupied_collision=bool(response.occupied_collision),
            unknown_queries=int(response.unknown_queries),
            occupied_queries=int(response.occupied_queries),
            minimum_obstacle_clearance=(
                clearance if math.isfinite(clearance) else None),
            front_center=(response.front_center_x, response.front_center_y),
            rear_center=(response.rear_center_x, response.rear_center_y),
            status=response.status)

    def _next_safe_waypoint(self, path):
        sparse = execution_waypoints(path, self._waypoint_spacing)
        if not sparse:
            return None, "empty_path"
        with self._lock:
            pose = self._pose
        # The first FAR marker vertex is normally the live robot pose.  It is
        # not a new command and must not be collision-validated as one: near a
        # narrow doorway the voxel map can contain self returns or wall noise
        # at the already occupied footprint.  Validating it rejected every
        # otherwise-safe forward suffix.  Drop only that reached vertex; the
        # next point (at most one waypoint spacing away) and every subsequent
        # segment remain checked by SCAN-lite.
        if (pose is not None and len(sparse) >= 2 and
                _distance(sparse[0], pose) <=
                max(0.10, 0.55 * self._waypoint_spacing)):
            sparse = sparse[1:]
        # SCAN-lite is a local execution guard.  Refining the complete FAR
        # global path caused hundreds/thousands of synchronous voxel service
        # calls and starved subsequent TARE goals.  Keep only the executable
        # local prefix plus one boundary point.
        local_sparse = [sparse[0]]
        travelled = 0.0
        for point in sparse[1:]:
            travelled += _distance(local_sparse[-1], point)
            local_sparse.append(point)
            if travelled >= self._scan_horizon_distance:
                break
        result = PathRefiner(
            self._scan_config, self._scan_check).refine(local_sparse)
        if not result.success:
            return None, "scan_lite_" + result.reason
        if pose is None:
            return None, "pose_unavailable"
        travelled = 0.0
        previous = (pose[0], pose[1])
        selected = result.path[-1]
        selected_yaw = result.yaws[-1]
        for point, yaw in zip(result.path, result.yaws):
            travelled += _distance(previous, point)
            selected, selected_yaw = point, yaw
            if travelled >= self._lookahead_distance:
                break
            previous = point
        if _distance(selected, pose) < self._minimum_execution_progress:
            # FAR may publish a partial graph path whose endpoint is within
            # far_goal_match_distance of the requested TARE goal.  Once that
            # partial endpoint has already been reached, executing it again
            # blocks this worker until the long goal timeout.  Treat it as a
            # consumed prefix so the caller can try the full A* fallback and
            # then request a fresh FAR path.
            return None, "path_prefix_already_reached"
        return (selected, selected_yaw), "scan_lite_safe"

    def _execute_waypoint(self, point, yaw, generation):
        message = PoseStamped()
        message.header.stamp = rospy.Time.now()
        message.header.frame_id = self._pose_frame
        message.pose.position.x, message.pose.position.y = point
        with self._condition:
            message.pose.position.z = self._pose[2] if self._pose else 0.0
            baseline = len(self._execution_results)
        message.pose.orientation.z = math.sin(yaw * 0.5)
        message.pose.orientation.w = math.cos(yaw * 0.5)
        self._goal_pub.publish(message)
        remaining = (
            self._maximum_duration
            if self._started is None else
            max(0.0, self._maximum_duration -
                (time.monotonic() - self._started)))
        deadline = time.monotonic() + min(self._goal_timeout, remaining)
        with self._condition:
            while not rospy.is_shutdown() and time.monotonic() < deadline:
                if len(self._execution_results) > baseline:
                    return dict(self._execution_results[-1])
                if (self._preempt_on_tare_goal_change and
                        generation != self._goal_generation):
                    self._cancel_pub.publish(String(data=json.dumps({
                        "reason": "tare_goal_superseded"})))
                    return {"success": False, "reason": "tare_goal_superseded"}
                self._condition.wait(timeout=0.10)
        time_limit = (
            self._started is not None and
            time.monotonic() - self._started >= self._maximum_duration)
        reason = "TIME_LIMIT" if time_limit else "goal_executor_timeout"
        self._cancel_pub.publish(String(data=json.dumps({"reason": reason})))
        return {"success": False, "reason": reason}

    def _run(self):
        while not rospy.is_shutdown():
            timed_out = False
            with self._condition:
                while (not rospy.is_shutdown() and
                       (self._pose is None or self._grid is None or
                        not self._locomotion_ready or
                        self._latest_tare_goal is None)):
                    if (time.monotonic() - self._started >=
                            self._maximum_duration):
                        timed_out = True
                        break
                    self._condition.wait(timeout=0.25)
                if rospy.is_shutdown():
                    return
                if timed_out:
                    goal = None
                    generation = self._goal_generation
                else:
                    goal = tuple(self._latest_tare_goal)
                    generation = self._goal_generation
            if timed_out:
                self._finish("time_limit")
                return
            if time.monotonic() - self._started >= self._maximum_duration:
                self._finish("time_limit")
                return
            with self._lock:
                home_reached = (
                    self._tare_finished and self._home_pose is not None and
                    _distance(self._pose, self._home_pose) <= 0.50)
            if home_reached:
                self._finish("tare_return_home_complete")
                return

            path, backend = self._request_far(goal, generation)
            if backend == "tare_goal_superseded":
                continue
            if path is None:
                path, backend = self._fallback_astar(goal)
            if path is None:
                self._status("planning_failed", backend=backend,
                             generation=generation)
                time.sleep(0.5)
                continue
            waypoint, safety = self._next_safe_waypoint(path)
            if waypoint is None and backend == "far":
                # A geometrically sparse FAR marker may expose a segment that
                # SCAN-lite proves unsafe (for example, a straight segment
                # through a room wall).  This is a FAR planning failure, so
                # use the documented global A* fallback to find the doorway.
                fallback_path, fallback_backend = self._fallback_astar(goal)
                if fallback_path is not None:
                    fallback_waypoint, fallback_safety = (
                        self._next_safe_waypoint(fallback_path))
                    if fallback_waypoint is not None:
                        path = fallback_path
                        backend = fallback_backend
                        waypoint = fallback_waypoint
                        safety = fallback_safety
            if waypoint is None:
                self._status("scan_rejected_path", reason=safety,
                             backend=backend, generation=generation)
                time.sleep(0.5)
                continue
            self._status("executing", backend=backend, safety=safety,
                         generation=generation)
            execution_record = {
                "cycle": len(self._execution_goal_history) + 1,
                "elapsed_sec": time.monotonic() - self._started,
                "goal": [waypoint[0][0], waypoint[0][1]],
                "tare_goal": list(goal),
                "tare_generation": generation,
                "backend": backend,
                "safety": safety,
                "success": None,
                "reason": None,
            }
            with self._lock:
                self._execution_goal_history.append(execution_record)
            outcome = self._execute_waypoint(
                waypoint[0], waypoint[1], generation)
            with self._lock:
                execution_record["success"] = bool(outcome.get("success"))
                execution_record["reason"] = str(outcome.get("reason", ""))
            self._status(
                "execution_result", backend=backend, generation=generation,
                success=bool(outcome.get("success")),
                reason=str(outcome.get("reason", "")))

    def _finish(self, reason):
        with self._condition:
            if self._finished:
                return
            self._finished = True
        self._status("finished", reason=reason)
        self._finalize_pub.publish(Bool(data=True))
        self._complete_pub.publish(Bool(data=True))
        rospy.signal_shutdown(reason)

    def _write_summary(self):
        data = {
            "architecture": "official_tare_official_far",
            "far_path_source": "/far/viz_path_topic",
            "fallback": "astar_only_after_far_failure",
            "finished": self._finished,
            "elapsed_sec": (
                time.monotonic() - self._started),
            "tare_goal_generation": self._goal_generation,
            "preempt_on_tare_goal_change": (
                self._preempt_on_tare_goal_change),
            "tare_finished": self._tare_finished,
            "latest_tare_goal": (
                list(self._latest_tare_goal)
                if self._latest_tare_goal is not None else None),
        }
        path = os.path.join(
            self._output_dir, "logs", "tare_far_summary.json")
        try:
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(data, stream, indent=2, sort_keys=True)
        except OSError as error:
            rospy.logwarn("Could not save TARE/FAR summary: %s", error)
        with self._lock:
            tare_history = list(self._tare_goal_history)
            execution_history = list(self._execution_goal_history)
        artifacts = (
            (os.path.join(
                self._output_dir, "logs", "tare_goal_history.json"),
             {"schema": "simenv_tare_goal_history_v1",
              "goals": tare_history}),
            (os.path.join(
                self._output_dir, "exploration_goal_history.json"),
             execution_history),
        )
        for artifact_path, artifact_data in artifacts:
            try:
                with open(
                        artifact_path, "w", encoding="utf-8") as stream:
                    json.dump(
                        artifact_data, stream, indent=2, sort_keys=True)
            except OSError as error:
                rospy.logwarn(
                    "Could not save exploration goal history %s: %s",
                    artifact_path, error)
        self._generate_visualization()

    def _generate_visualization(self):
        """Start diagnostics independently of roslaunch's shutdown deadline."""
        if not self._auto_visualization:
            return
        if not os.path.isfile(self._visualization_script):
            rospy.logwarn(
                "Visualization script not found: %s",
                self._visualization_script)
            return
        if not os.path.isfile(self._visualization_launcher):
            rospy.logwarn(
                "Visualization launcher not found: %s",
                self._visualization_launcher)
            return
        log_path = os.path.join(
            self._output_dir, "visualization_generation.log")
        command = [
            sys.executable, self._visualization_launcher,
            "--run-dir", self._output_dir,
            "--visualizer", self._visualization_script,
        ]
        try:
            log_stream = open(log_path, "a", encoding="utf-8")
            subprocess.Popen(
                command, stdout=log_stream, stderr=subprocess.STDOUT,
                start_new_session=True, close_fds=True)
            log_stream.close()
        except OSError as error:
            rospy.logwarn("Automatic visualization failed: %s", error)
            return
        rospy.loginfo(
            "Detached automatic visualization started; output=%s",
            os.path.join(self._output_dir, "visualization"))


if __name__ == "__main__":
    rospy.init_node("far_interface")
    FarInterface()
    rospy.spin()
