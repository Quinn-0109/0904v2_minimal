#!/usr/bin/env python3
"""Hierarchical graph goal selection and execution through existing interfaces."""

import csv
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import Bool, Int32, String

from baseline_planning_core import execution_waypoints
from exploration_graph import ExplorationGraph
from hierarchical_explorer import HierarchicalGoalSelector, SelectorConfig
from hierarchical_ros_utils import atomic_json, occupancy_from_message
from planner_interface import AStarPlanner, FarPlannerInterface
from scan_lite_path_refiner import FootprintCheck, PathRefiner, RefinerConfig
from simenv_competitor.srv import CheckTwinCylinder, CheckTwinCylinderRequest


class HierarchicalExplorerNode:
    def __init__(self):
        self.lock = threading.RLock()
        self.process_started = time.monotonic()
        self.started = None
        self.pose = None
        self.pose_frame = "camera_init"
        self.grid = None
        self.graph = None
        self.locomotion_ready = False
        self.execution_results = []
        self.mission_start_pose = None
        self.trajectory = []
        self.total_path_length = 0.0
        self.goal_count = 0
        self.consecutive_failures = 0
        self.goal_history = []
        self.open_visits = {}
        self.visit_rows = []
        self.finished = False
        self.map_saved_count = 0
        self.visualization_generated = False

        self.output_dir = rospy.get_param("~output_dir", "/tmp/simenv")
        self.log_dir = os.path.join(self.output_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.maximum_duration = float(rospy.get_param(
            "~maximum_duration", 700.0))
        self.maximum_goals = int(rospy.get_param("~maximum_goals", 100))
        self.startup_timeout = float(rospy.get_param(
            "~startup_timeout", 120.0))
        self.minimum_graph_updates = int(rospy.get_param(
            "~minimum_graph_updates", 2))
        self.goal_timeout = float(rospy.get_param("~goal_timeout", 180.0))
        self.waypoint_spacing = float(rospy.get_param(
            "~waypoint_spacing", 0.45))
        self.goal_topic = rospy.get_param(
            "~goal_topic", "/exploration_goal")
        self.planner_name = str(rospy.get_param(
            "~planner", "astar")).lower()
        self.auto_generate_visualization = bool(rospy.get_param(
            "~auto_generate_visualization", True))
        default_visualizer = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "scripts",
            "visualize_baseline_results.py"))
        self.visualization_script = os.path.abspath(rospy.get_param(
            "~visualization_script", default_visualizer))
        self.spawn_pose = {
            "x": float(rospy.get_param("~robot_spawn_x", 0.0)),
            "y": float(rospy.get_param("~robot_spawn_y", 0.8)),
            "z": float(rospy.get_param("~robot_spawn_z", 0.6)),
            "yaw": float(rospy.get_param("~robot_spawn_yaw", 1.5708)),
            "sensor_forward_offset": float(rospy.get_param(
                "~offline_truth_sensor_forward_offset", 0.20)),
        }
        truth_layout = os.path.abspath(rospy.get_param(
            "~offline_truth_layout_metadata", ""))
        if truth_layout and os.path.isfile(truth_layout):
            try:
                shutil.copy2(
                    truth_layout,
                    os.path.join(self.output_dir, "layout_metadata.json"))
            except OSError as error:
                rospy.logwarn(
                    "Could not snapshot offline truth layout: %s", error)

        astar = AStarPlanner(
            clearance=rospy.get_param("~astar_clearance", 0.30),
            reached_tolerance=rospy.get_param(
                "~reached_tolerance", 0.15))
        self.planner = (
            FarPlannerInterface(fallback=astar)
            if self.planner_name == "far" else astar)
        self.selector = HierarchicalGoalSelector(
            self.planner,
            SelectorConfig(
                alpha=rospy.get_param("~alpha", 1.0),
                beta=rospy.get_param("~beta", 0.5),
                gamma=rospy.get_param("~gamma", 1.5),
                revisit_lambda=rospy.get_param("~lambda", 1.0),
                coverage_threshold=rospy.get_param(
                    "~coverage_threshold", 0.90),
                unknown_ratio_threshold=rospy.get_param(
                    "~unknown_ratio_threshold", 0.05),
                local_coverage_radius=rospy.get_param(
                    "~local_coverage_radius", 6.0),
                region_enter_tolerance=rospy.get_param(
                    "~region_enter_tolerance", 1.5),
                entry_return_tolerance=rospy.get_param(
                    "~entry_return_tolerance", 0.50),
                minimum_information_gain=rospy.get_param(
                    "~minimum_information_gain", 0.20),
                minimum_new_region_seconds=rospy.get_param(
                    "~minimum_new_region_seconds", 35.0),
                nominal_speed=rospy.get_param("~nominal_speed", 0.30),
                recent_goal_distance=rospy.get_param(
                    "~recent_goal_distance", 0.75),
                failure_cooldown_seconds=rospy.get_param(
                    "~failure_cooldown_seconds", 30.0),
            ))
        self.scan_config = RefinerConfig(
            unknown_policy=rospy.get_param(
                "~scan_lite_unknown_policy", "penalize"),
            body_safety_margin=rospy.get_param(
                "~scan_lite_body_safety_margin", 0.05),
            refinement_search_radius=rospy.get_param(
                "~scan_lite_search_radius", 0.60),
            refinement_search_step=rospy.get_param(
                "~scan_lite_search_step", 0.15),
        )

        self.goal_pub = rospy.Publisher(
            self.goal_topic, PoseStamped, queue_size=1)
        self.state_pub = rospy.Publisher(
            "/simenv/hierarchical_exploration_state",
            String, queue_size=2, latch=True)
        self.visit_pub = rospy.Publisher(
            "/simenv/region_visit_update", String, queue_size=10)
        self.active_goal_pub = rospy.Publisher(
            "/simenv/active_goal_id", Int32, queue_size=2, latch=True)
        self.active_waypoint_pub = rospy.Publisher(
            "/simenv/active_waypoint_id", Int32, queue_size=2, latch=True)
        self.cancel_pub = rospy.Publisher(
            "/simenv/cancel_exploration_goal", String, queue_size=2)
        self.finalize_pub = rospy.Publisher(
            "/simenv/finalize_voxel_map", Bool, queue_size=1, latch=True)
        self.complete_pub = rospy.Publisher(
            "/simenv/mission_complete", Bool, queue_size=1, latch=True)
        service_name = rospy.get_param(
            "~scan_service",
            "/hierarchical_voxel_mapper/check_twin_cylinder")
        self.scan_service = rospy.ServiceProxy(
            service_name, CheckTwinCylinder)

        rospy.Subscriber(
            rospy.get_param("~odom_topic", "/Odometry"),
            Odometry, self._on_odom, queue_size=30)
        rospy.Subscriber(
            rospy.get_param(
                "~map_topic", "/simenv/voxel_floor_projection"),
            OccupancyGrid, self._on_map, queue_size=1)
        rospy.Subscriber(
            "/simenv/exploration_graph", String,
            self._on_graph, queue_size=1)
        rospy.Subscriber(
            "/simenv/goal_execution_result", String,
            self._on_execution_result, queue_size=20)
        rospy.Subscriber(
            "/locomotion_ready", Bool,
            self._on_locomotion, queue_size=2)
        rospy.Subscriber(
            "/simenv/voxel_map_saved", Bool,
            self._on_map_saved, queue_size=2)
        rospy.on_shutdown(self._write_logs)
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def elapsed(self):
        return (
            time.monotonic() - self.started
            if self.started is not None else 0.0)

    def startup_elapsed(self):
        return time.monotonic() - self.process_started

    def _state(self, state, reason=""):
        payload = {
            "state": state,
            "reason": reason,
            "elapsed_sec": round(self.elapsed(), 3),
            "goal_count": self.goal_count,
            "selector": self.selector.state(),
        }
        self.state_pub.publish(String(data=json.dumps(payload)))

    def _on_locomotion(self, message):
        self.locomotion_ready = bool(message.data)

    def _on_map_saved(self, message):
        if message.data:
            with self.lock:
                self.map_saved_count += 1

    def _on_odom(self, message):
        point = message.pose.pose.position
        pose = (float(point.x), float(point.y), float(point.z))
        if not all(math.isfinite(value) for value in pose):
            return
        with self.lock:
            previous = self.pose
            self.pose = pose
            self.pose_frame = message.header.frame_id or self.pose_frame
            if self.mission_start_pose is None:
                self.mission_start_pose = pose
            if previous is not None:
                self.total_path_length += math.hypot(
                    pose[0] - previous[0], pose[1] - previous[1])
            now = self.elapsed()
            if not self.trajectory or now - self.trajectory[-1][0] >= 0.10:
                self.trajectory.append(
                    (now, pose[0], pose[1], pose[2]))

    def _on_map(self, message):
        try:
            grid = occupancy_from_message(message)
        except (ValueError, IndexError):
            return
        with self.lock:
            self.grid = grid

    def _on_graph(self, message):
        try:
            graph = ExplorationGraph.from_dict(json.loads(message.data))
        except (KeyError, TypeError, ValueError):
            return
        with self.lock:
            self.graph = graph

    def _on_execution_result(self, message):
        try:
            result = json.loads(message.data)
        except (TypeError, ValueError):
            return
        with self.lock:
            self.execution_results.append(result)

    def _scan_check(self, x, y, yaw):
        request = CheckTwinCylinderRequest()
        request.pose_x, request.pose_y = x, y
        with self.lock:
            request.pose_z = self.pose[2] if self.pose else 0.0
        request.yaw = yaw
        request.front_offset = self.scan_config.body_front_offset
        request.rear_offset = self.scan_config.body_rear_offset
        request.radius = (
            self.scan_config.body_collision_radius +
            self.scan_config.body_safety_margin)
        request.min_height = (
            self.scan_config.body_min_height -
            self.scan_config.body_safety_margin)
        request.max_height = (
            self.scan_config.body_max_height +
            self.scan_config.body_safety_margin)
        request.clearance_search_radius = request.radius + 0.50
        try:
            response = self.scan_service(request)
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
            front_center=(
                response.front_center_x, response.front_center_y),
            rear_center=(
                response.rear_center_x, response.rear_center_y),
            status=response.status)

    def _safe_execution_path(self, path):
        sparse = execution_waypoints(path, self.waypoint_spacing)
        if not sparse:
            return None, "empty_path"
        refinement = PathRefiner(
            self.scan_config, self._scan_check).refine(sparse)
        if not refinement.success:
            return None, "scan_lite_" + refinement.reason
        return list(zip(refinement.path, refinement.yaws)), "safe"

    def _recent_backtrack(self, maximum_distance=3.0):
        """Reverse poses the robot physically occupied when SCAN rejects exit."""
        with self.lock:
            samples = list(self.trajectory)
            pose = self.pose
        if pose is None or len(samples) < 2:
            return []
        selected = []
        travelled = 0.0
        previous = (pose[0], pose[1])
        for sample in reversed(samples[:-1]):
            point = (float(sample[1]), float(sample[2]))
            step = math.hypot(
                point[0] - previous[0], point[1] - previous[1])
            travelled += step
            if (not selected or math.hypot(
                    point[0] - selected[-1][0],
                    point[1] - selected[-1][1]) >=
                    self.waypoint_spacing):
                selected.append(point)
            previous = point
            if travelled >= maximum_distance:
                break
        output = []
        current = (pose[0], pose[1])
        for point in selected:
            yaw = math.atan2(
                point[1] - current[1], point[0] - current[0])
            output.append((point, yaw))
            current = point
        return output

    def _publish_waypoint(self, point, yaw):
        message = PoseStamped()
        message.header.stamp = rospy.Time.now()
        message.header.frame_id = self.pose_frame
        message.pose.position.x = float(point[0])
        message.pose.position.y = float(point[1])
        with self.lock:
            message.pose.position.z = self.pose[2] if self.pose else 0.0
            baseline = len(self.execution_results)
        message.pose.orientation.z = math.sin(yaw * 0.5)
        message.pose.orientation.w = math.cos(yaw * 0.5)
        self.goal_pub.publish(message)
        deadline = time.monotonic() + min(
            self.goal_timeout,
            max(0.0, self.maximum_duration - self.elapsed()))
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            with self.lock:
                if len(self.execution_results) > baseline:
                    return dict(self.execution_results[-1])
            time.sleep(0.05)
        reason = (
            "TIME_LIMIT" if self.elapsed() >= self.maximum_duration
            else "goal_executor_timeout")
        self.cancel_pub.publish(String(data=json.dumps({
            "goal_stamp": message.header.stamp.to_sec(),
            "goal_x": point[0], "goal_y": point[1],
            "reason": reason,
        })))
        return {"success": False, "reason": reason}

    def _execute(self, cycle, decision):
        result = decision.path_result
        if result is None or not result.success:
            return False, (
                result.reason if result is not None else "no_path")
        refined, reason = self._safe_execution_path(result.path)
        if refined is None:
            if decision.kind != "return_region_entry":
                return False, reason
            backtrack = self._recent_backtrack()
            if not backtrack:
                return False, reason
            for point, yaw in backtrack:
                outcome = self._publish_waypoint(point, yaw)
                if not outcome.get("success"):
                    return False, reason + "_backtrack_failed"
            return False, reason + "_verified_backtrack_completed"
        self.active_goal_pub.publish(Int32(data=cycle))
        for index, (point, yaw) in enumerate(refined):
            self.active_waypoint_pub.publish(Int32(data=index))
            outcome = self._publish_waypoint(point, yaw)
            if not outcome.get("success"):
                self.active_goal_pub.publish(Int32(data=-1))
                self.active_waypoint_pub.publish(Int32(data=-1))
                return False, str(outcome.get(
                    "reason", "executor_failure"))
        self.active_goal_pub.publish(Int32(data=-1))
        self.active_waypoint_pub.publish(Int32(data=-1))
        return True, "goal_reached"

    def _publish_visit(self, update):
        if not update.get("region_id"):
            return
        self.visit_pub.publish(String(data=json.dumps(update)))
        region_id = update["region_id"]
        if update.get("entered"):
            self.open_visits[region_id] = {
                "region_id": region_id,
                "enter_time": update["time"],
                "coverage_before": update.get("coverage_ratio", 0.0),
            }
        if update.get("exited"):
            row = self.open_visits.pop(region_id, {
                "region_id": region_id,
                "enter_time": "",
                "coverage_before": "",
            })
            row["exit_time"] = update["time"]
            row["coverage_after"] = update.get("coverage_ratio", 0.0)
            self.visit_rows.append(row)

    def _write_logs(self):
        payload = {
            "schema": "simenv_hierarchical_exploration_v1",
            "elapsed_sec": self.elapsed(),
            "maximum_duration": self.maximum_duration,
            "goal_count": self.goal_count,
            "total_path_length": self.total_path_length,
            "planner": self.planner_name,
            "selector_state": self.selector.state(),
            "goals": self.goal_history,
        }
        atomic_json(os.path.join(
            self.log_dir, "hierarchical_goal_history.json"), payload)
        atomic_json(os.path.join(
            self.output_dir, "hierarchical_summary.json"), {
                key: value for key, value in payload.items()
                if key != "goals"})
        atomic_json(os.path.join(
            self.output_dir, "startup_status.json"), {
                "schema": "simenv_hierarchical_startup_status_v1",
                "elapsed_sec": self.startup_elapsed(),
                "exploration_end_elapsed_sec": self.startup_elapsed(),
                "exploration_duration_sec": self.elapsed(),
                "robot_spawn_pose": self.spawn_pose,
                "locomotion_ready": self.locomotion_ready,
            })
        csv_path = os.path.join(self.log_dir, "region_visit.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=[
                "region_id", "enter_time", "exit_time",
                "coverage_before", "coverage_after"])
            writer.writeheader()
            writer.writerows(self.visit_rows)

    def _generate_visualization(self):
        if (not self.auto_generate_visualization or
                self.visualization_generated):
            return
        self.visualization_generated = True
        if not os.path.isfile(self.visualization_script):
            rospy.logerr(
                "Visualization script missing: %s",
                self.visualization_script)
            return
        command = [
            sys.executable, self.visualization_script,
            "--run-dir", self.output_dir,
            "--output-dir",
            os.path.join(self.output_dir, "visualization"),
        ]
        try:
            completed = subprocess.run(
                command, timeout=120.0, check=False)
            if completed.returncode:
                rospy.logerr(
                    "Hierarchical visualization exited with code %d",
                    completed.returncode)
            else:
                rospy.loginfo(
                    "Hierarchical visualization saved under %s/visualization",
                    self.output_dir)
        except (OSError, subprocess.TimeoutExpired) as error:
            rospy.logerr(
                "Hierarchical visualization failed: %s", error)

    def _finish(self, reason):
        if self.finished:
            return
        self.finished = True
        self._state("RETURN_HOME_COMPLETE", reason)
        self._write_logs()
        with self.lock:
            saved_before = self.map_saved_count
        self.finalize_pub.publish(Bool(data=True))
        self.complete_pub.publish(Bool(data=True))
        deadline = time.monotonic() + 5.0
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            with self.lock:
                if self.map_saved_count > saved_before:
                    break
            time.sleep(0.05)
        self._write_logs()
        self._generate_visualization()
        rospy.signal_shutdown(reason)

    def _run(self):
        rate = rospy.Rate(2.0)
        while not rospy.is_shutdown():
            with self.lock:
                ready = (
                    self.pose is not None and self.grid is not None and
                    self.graph is not None and
                    self.graph.update_sequence >=
                    self.minimum_graph_updates and
                    any(node.type in ("region", "visited")
                        for node in self.graph.nodes.values()) and
                    self.locomotion_ready)
            if ready:
                self.started = time.monotonic()
                break
            if self.startup_elapsed() >= self.startup_timeout:
                self._finish("STARTUP_TIMEOUT")
                return
            self._state("WAITING_FOR_INPUTS")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                return
        while not rospy.is_shutdown() and not self.finished:
            with self.lock:
                pose = self.pose
                grid = self.grid
                graph = self.graph
                start = self.mission_start_pose
            remaining = self.maximum_duration - self.elapsed()
            if remaining <= 0.0 or self.goal_count >= self.maximum_goals:
                decision = None
                reason = (
                    "TIME_LIMIT" if remaining <= 0.0 else "GOAL_LIMIT")
            else:
                decision = self.selector.select(
                    graph, pose[:2], grid, remaining)
                reason = decision.reason
            if decision is None or decision.kind == "return_home":
                if start is None:
                    self._finish(reason)
                    return
                home = self.planner.plan(pose[:2], start[:2], grid)
                if (not home.success or math.hypot(
                        pose[0] - start[0], pose[1] - start[1]) <= 0.5):
                    self._finish(reason)
                    return
                from hierarchical_explorer import GoalDecision
                decision = GoalDecision(
                    kind="return_home", goal=start[:2], reason=reason,
                    path_result=home, travel_cost=home.travel_cost)
            elif decision.kind == "region_complete":
                update = {
                    "region_id": decision.region_id,
                    "coverage_ratio": decision.coverage_ratio,
                    "entered": False, "exited": True,
                    "success": True,
                    "time": rospy.Time.now().to_sec(),
                }
                self._publish_visit(update)
                time.sleep(0.1)
                continue

            self.goal_count += 1
            cycle = self.goal_count
            self._state("EXECUTE_" + decision.kind.upper(), decision.reason)
            start_pose = pose[:2]
            success, execution_reason = self._execute(cycle, decision)
            with self.lock:
                final_pose = self.pose
            now = rospy.Time.now().to_sec()
            update = self.selector.record_goal_result(
                decision, success, final_pose[:2], now,
                start_pose=start_pose)
            self._publish_visit(update)
            self.goal_history.append({
                "cycle": cycle,
                "elapsed_sec": self.elapsed(),
                "decision": decision.to_dict(),
                "success": success,
                "execution_reason": execution_reason,
                "start_pose": list(start_pose),
                "final_pose": list(final_pose[:2]),
            })
            self._write_logs()
            if success:
                self.consecutive_failures = 0
            else:
                self.consecutive_failures += 1
            if decision.kind == "return_home":
                self._finish(decision.reason)
                return
            if self.consecutive_failures >= 5:
                self._finish("CONSECUTIVE_FAILURE_LIMIT")
                return
            time.sleep(0.15)


if __name__ == "__main__":
    rospy.init_node("hierarchical_explorer_node")
    HierarchicalExplorerNode()
    rospy.spin()
