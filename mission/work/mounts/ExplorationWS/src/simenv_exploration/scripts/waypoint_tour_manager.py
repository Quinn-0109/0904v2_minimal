#!/usr/bin/env python3
"""Serial fixed-waypoint tour driver for whole-flow testing.

Replaces the per-floor free-exploration managers with a fixed list of
waypoints executed serially through the Goal Executor, so the floor-switching
and hazard-detection chain can be exercised without frontier exploration:

- F1: a standalone instance of this node (launch ``waypoint_tour_mode:=true``)
  gates on FAST-LIO odometry + RL locomotion, drives the waypoints, then
  publishes STAIR_WAIT_ZONE on /simenv/baseline_state.  The stair manager only
  matches that substring and takes over with its own truth-guided approach,
  so stair_transition_manager.py is untouched.
- F2/F3: SecondFloorMission imports WaypointTourRunner and replaces
  BaselineExplorationManager().run() with it while ``~tour_mode`` is set; all
  existing token handoff (SECOND_FLOOR_EXPLORATION_COMPLETE etc.) is kept.

Free exploration remains the default; this file only activates when the
launch selects the tour mode.  Unlike the free-exploration managers, no
``/simenv/finalize_result`` or ``/simenv/goal_executor_pause`` messages are
ever published here (the stair manager ignores the former once it leaves
WAIT_F1, and nothing would clear the latter).
"""

import json
import math
import os
import sys
import threading
import time

# A catkin-installed Python executable is a relay under devel/lib.  Without
# explicitly preferring this source directory, the sibling import below finds
# the relay for baseline_planning_core instead of the actual module.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if not sys.path or sys.path[0] != _SCRIPT_DIR:
    sys.path.insert(0, _SCRIPT_DIR)

import rospy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float32, String

from baseline_planning_core import execution_result_matches


class WaypointTourRunner:
    """Serial waypoint executor shared by the F1 node and the F2/F3 missions.

    Each waypoint is published as a PoseStamped goal for the Goal Executor
    (``/exploration_goal``) and awaited via ``/simenv/goal_execution_result``,
    matched with the same stamp/position contract the free-exploration manager
    uses (``execution_result_matches``).  A per-waypoint timeout cancels the
    goal and fails the whole tour.
    """

    SUMMARY_SCHEMA = "simenv_waypoint_tour_v1"

    def __init__(self, floor_number=1, floor_slug="first_floor",
                 output_dir=None, waypoints=None,
                 odom_topic="/simenv/fastlio_stabilized_odometry",
                 default_speed=1.0, default_lateral_speed=0.35,
                 waypoint_timeout=60.0, startup_timeout=120.0,
                 wait_startup=True, publish_baseline_state=False,
                 waypoint_frame="world",
                 ground_truth_topic="/gazebo/model_states",
                 ground_truth_model="a1_gazebo",
                 world_origin=None):
        self._floor_number = int(floor_number)
        self._floor_slug = str(floor_slug)
        self._output_dir = str(output_dir) if output_dir else ""
        self.waypoints = (list(waypoints) if waypoints else [])
        self._default_speed = float(default_speed)
        self._default_lateral_speed = float(default_lateral_speed)
        self._waypoint_timeout = float(waypoint_timeout)
        self._startup_timeout = float(startup_timeout)
        self._wait_startup = bool(wait_startup)
        self._publish_baseline_state = bool(publish_baseline_state)
        # Tour waypoints are authored in world coordinates (they mirror the
        # fixed stair/corridor layout), but the Goal Executor navigates in
        # the FAST-LIO camera_init frame: the robot's initial body pose,
        # rotated ~90 deg from world because the dog spawns facing +y.  The
        # free-exploration manager is self-consistent inside camera_init and
        # the stair manager drives its own truth route, so the tour is the
        # only caller that must bridge frames.  The origin is read from
        # Gazebo truth (same source as the stair manager) unless an explicit
        # [x, y, yaw] override is supplied.
        self._waypoint_frame = str(waypoint_frame)
        self._ground_truth_model = str(ground_truth_model)
        self._world_origin = None
        # The camera_init anchor is fixed at FAST-LIO's startup (the F1 spawn
        # pose), so every floor must use that same origin.  The F1 runner
        # publishes its resolved anchor on a latched topic; F2/F3 runners
        # prefer it over their own first truth sample, which would otherwise
        # be the (wrong) pose they happen to latch at their construction.
        self._local_origin = None
        self._shared_origin = None
        if self._waypoint_frame == "world":
            if world_origin is not None:
                self._world_origin = [float(v) for v in world_origin]
            else:
                rospy.Subscriber(str(ground_truth_topic), ModelStates,
                                 self._on_ground_truth, queue_size=2)
                self._origin_pub = (
                    rospy.Publisher("/simenv/tour_world_origin", String,
                                    queue_size=1, latch=True)
                    if publish_baseline_state else None)
                if self._origin_pub is None:
                    rospy.Subscriber("/simenv/tour_world_origin", String,
                                     self._on_shared_origin, queue_size=2)
        self._lock = threading.RLock()
        self._pose = None
        self._pose_frame = "camera_init"
        self._locomotion_ready = False
        self._execution_results = []
        self.results = []
        self.termination_reason = None
        self._started = time.monotonic()
        self._started_wall_time = time.time()
        self._goal_pub = rospy.Publisher(
            "/exploration_goal", PoseStamped, queue_size=1)
        self._speed_pub = rospy.Publisher(
            "/simenv/goal_speed_limit", Float32, queue_size=1, latch=True)
        self._lateral_pub = rospy.Publisher(
            "/simenv/goal_lateral_speed_limit", Float32, queue_size=1,
            latch=True)
        self._cancel_pub = rospy.Publisher(
            "/simenv/cancel_exploration_goal", String, queue_size=2)
        self._state_pub = None
        if self._publish_baseline_state:
            self._state_pub = rospy.Publisher(
                "/simenv/baseline_state", String, queue_size=1, latch=True)
        rospy.Subscriber("/simenv/goal_execution_result", String,
                         self._on_execution_result, queue_size=10)
        rospy.Subscriber(odom_topic, Odometry, self._on_odom, queue_size=20)
        rospy.Subscriber("/locomotion_ready", Bool,
                         self._on_locomotion_ready, queue_size=2)

    # ------------------------------------------------------------------ gating
    def wait_for_startup(self):
        """F1-equivalent gate: FAST-LIO odometry plus RL locomotion ready.

        The free-exploration manager additionally waits for a non-empty grid;
        the tour only needs stable odometry because the Goal Executor gates
        every goal on the same readiness topics.
        """
        if not self._wait_startup:
            return True
        deadline = time.monotonic() + self._startup_timeout
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            with self._lock:
                ready = self._pose is not None and self._locomotion_ready
            if ready:
                return True
            time.sleep(0.1)
        return False

    # ------------------------------------------------------------------- tour
    def run(self):
        """Execute the waypoint list serially.  Returns True iff every
        waypoint was reached; the summary is rewritten after each step."""
        self._started = time.monotonic()
        self._started_wall_time = time.time()
        self.results = []
        self.termination_reason = None
        if self._waypoint_frame == "world" and self._world_origin is None:
            deadline = time.monotonic() + self._startup_timeout
            local_fallback_after = time.monotonic() + 15.0
            while (not rospy.is_shutdown() and time.monotonic() < deadline
                   and self._world_origin is None):
                if self._origin_pub is not None:
                    # F1: camera_init is anchored at our own first truth
                    # sample (the spawn pose before any motion).
                    if self._local_origin is not None:
                        self._world_origin = self._local_origin
                elif self._shared_origin is not None:
                    # F2/F3: inherit the F1-published anchor (latched, so it
                    # arrives on connect regardless of when F1 resolved it).
                    self._world_origin = self._shared_origin
                elif (self._local_origin is not None
                      and time.monotonic() >= local_fallback_after):
                    rospy.logwarn(
                        "[TOUR] no shared world origin from F1; falling back "
                        "to the local truth sample (%.3f, %.3f, yaw %.3f) "
                        "-- only valid if camera_init was anchored there",
                        *self._local_origin)
                    self._world_origin = self._local_origin
                if self._world_origin is None:
                    time.sleep(0.1)
            if self._world_origin is None:
                self.termination_reason = "WORLD_ORIGIN_UNAVAILABLE"
                rospy.logerr(
                    "[TOUR] waypoint_frame=world but no Gazebo truth for %s "
                    "on %s within %.0f s; cannot map waypoints into the "
                    "executor frame", self._ground_truth_model,
                    "the origin override", self._startup_timeout)
                return False
            if self._origin_pub is not None:
                self._origin_pub.publish(String(
                    data=json.dumps(self._world_origin, sort_keys=True)))
            rospy.loginfo("[TOUR] world origin %s -> camera_init: "
                          "x0=%.3f y0=%.3f yaw0=%.3f", self._ground_truth_model,
                          *self._world_origin)
        for index, waypoint in enumerate(self.waypoints):
            success, record = self._execute_waypoint(index, waypoint)
            self.results.append(record)
            self._write_summary()
            if not success:
                self.termination_reason = "WAYPOINT_{}_FAILED:{}".format(
                    index, record.get("executor_reason", "timeout"))
                return False
        self.termination_reason = "WAYPOINT_TOUR_COMPLETE"
        self._write_summary()
        return True

    def _execute_waypoint(self, index, waypoint):
        world_point = (float(waypoint["x"]), float(waypoint["y"]))
        world_yaw = float(waypoint["yaw"])
        # Map world-authored waypoints into the executor's camera_init frame
        # so the odometry-based navigation sees the intended world pose.
        if self._world_origin is not None:
            x0, y0, yaw0 = self._world_origin
            dx, dy = world_point[0] - x0, world_point[1] - y0
            cosine, sine = math.cos(yaw0), math.sin(yaw0)
            point = (cosine * dx + sine * dy, -sine * dx + cosine * dy)
            yaw = world_yaw - yaw0
        else:
            point, yaw = world_point, world_yaw
        # Normalized waypoints always carry the keys (possibly None), so a
        # bare .get(key, default) would return None and float() would crash.
        raw_speed = waypoint.get("speed")
        speed = max(0.05, float(
            raw_speed if raw_speed is not None else self._default_speed))
        self._speed_pub.publish(Float32(data=speed))
        self._lateral_pub.publish(Float32(data=self._default_lateral_speed))
        message = PoseStamped()
        message.header.stamp = rospy.Time.now()
        message.header.frame_id = self._pose_frame
        message.pose.position.x = point[0]
        message.pose.position.y = point[1]
        message.pose.position.z = float(waypoint["z"])
        message.pose.orientation.z = math.sin(yaw * 0.5)
        message.pose.orientation.w = math.cos(yaw * 0.5)
        goal_stamp = float(message.header.stamp.to_sec())
        with self._lock:
            baseline = len(self._execution_results)
        self._goal_pub.publish(message)
        rospy.loginfo(
            "[TOUR] floor %d published goal %d: stamp=%.4f point=(%.3f, "
            "%.3f, %.3f) yaw=%.3f frame=%s", self._floor_number, index,
            goal_stamp, point[0], point[1], float(waypoint["z"]), yaw,
            message.header.frame_id)
        raw_timeout = waypoint.get("timeout")
        timeout = float(raw_timeout if raw_timeout is not None
                        else self._waypoint_timeout)
        deadline = time.monotonic() + timeout
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            with self._lock:
                candidates = list(self._execution_results[baseline:])
            for candidate in candidates:
                if execution_result_matches(candidate, goal_stamp, point):
                    return (bool(candidate.get("success")),
                            self._record(index, waypoint, goal_stamp, candidate,
                                         point, yaw, world_point, world_yaw))
            time.sleep(0.05)
        # Per-waypoint timeout (kept below the executor's own goal_timeout so
        # the tour decides before the executor reports a terminal failure).
        # Cancel exactly this goal per the manager contract.
        self._cancel_pub.publish(String(data=json.dumps({
            "goal_sequence": int(message.header.seq),
            "goal_stamp": goal_stamp,
            "goal_x": point[0],
            "goal_y": point[1],
            "reason": "waypoint_tour_timeout",
        }, sort_keys=True)))
        record = {
            "index": index,
            "x": point[0],
            "y": point[1],
            "z": float(waypoint["z"]),
            "yaw": yaw,
            "world_x": world_point[0],
            "world_y": world_point[1],
            "world_yaw": world_yaw,
            "note": waypoint.get("note"),
            "goal_stamp": goal_stamp,
            "success": False,
            "executor_reason": "tour_timeout",
            "distance_error_m": None,
            "duration_sec": timeout,
        }
        return False, record

    @staticmethod
    def _record(index, waypoint, goal_stamp, candidate,
                point, yaw, world_point, world_yaw):
        return {
            "index": index,
            "x": point[0],
            "y": point[1],
            "z": float(waypoint["z"]),
            "yaw": yaw,
            "world_x": world_point[0],
            "world_y": world_point[1],
            "world_yaw": world_yaw,
            "note": waypoint.get("note"),
            "goal_stamp": goal_stamp,
            "success": bool(candidate.get("success")),
            "executor_reason": str(candidate.get("reason", "")),
            "distance_error_m": candidate.get("distance_error_m"),
            "duration_sec": candidate.get("duration_sec"),
        }

    # ---------------------------------------------------------------- helpers
    def publish_state(self, state, **fields):
        """Publish a baseline_state-style payload (F1 only).  The stair
        manager matches the state substring, so JSON field names are free."""
        if self._state_pub is None:
            return
        payload = {"state": state, "phase": "MISSION",
                   "elapsed_sec": round(time.monotonic() - self._started, 3)}
        payload.update(fields)
        self._state_pub.publish(String(
            data=json.dumps(payload, sort_keys=True)))

    def _write_summary(self):
        if not self._output_dir:
            return
        payload = {
            "schema": self.SUMMARY_SCHEMA,
            "floor_number": self._floor_number,
            "success": self.termination_reason == "WAYPOINT_TOUR_COMPLETE",
            "termination_reason": self.termination_reason,
            "started_wall_time": self._started_wall_time,
            "finished_wall_time": time.time(),
            "duration_sec": round(time.monotonic() - self._started, 3),
            "waypoint_frame": self._waypoint_frame,
            "world_origin": self._world_origin,
            "waypoints": self.results,
        }
        path = os.path.join(self._output_dir, "tour_summary.json")
        try:
            os.makedirs(self._output_dir, exist_ok=True)
            with open(path + ".tmp", "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False,
                          indent=2, sort_keys=True)
                stream.write("\n")
            os.replace(path + ".tmp", path)
        except OSError as error:
            rospy.logwarn_throttle(10.0, "[TOUR] summary write failed: %s",
                                   error)

    @staticmethod
    def _normalize_waypoints(raw):
        """Accept a JSON string (launch <param> attribute) or a parsed list."""
        if isinstance(raw, str):
            raw = json.loads(raw)
        if not isinstance(raw, list) or not raw:
            raise ValueError("waypoints must be a non-empty list")
        waypoints = []
        for item in raw:
            try:
                x = float(item["x"])
                y = float(item["y"])
                z = float(item.get("z", 0.0))
                yaw = float(item.get("yaw", 0.0))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    "invalid waypoint {!r}".format(item)) from error
            if not all(math.isfinite(value)
                       for value in (x, y, z, yaw)):
                raise ValueError(
                    "waypoint contains non-finite value: {!r}".format(item))
            waypoints.append({
                "x": x, "y": y, "z": z, "yaw": yaw,
                "speed": item.get("speed"),
                "timeout": item.get("timeout"),
                "note": item.get("note"),
            })
        return waypoints

    # -------------------------------------------------------------- callbacks
    def _on_odom(self, message):
        pose = message.pose.pose.position
        with self._lock:
            self._pose = (pose.x, pose.y, pose.z)
            if message.header.frame_id:
                self._pose_frame = message.header.frame_id

    def _on_ground_truth(self, message):
        """Remember the first model pose as a candidate camera_init origin.
        FAST-LIO anchors camera_init at the robot's initial body pose, so the
        inverse of that pose maps world waypoints into the executor's frame.
        Only the F1 runner's first sample is the true anchor; later samples
        are used as a fallback when no shared anchor is available."""
        if self._local_origin is not None:
            return
        try:
            index = message.name.index(self._ground_truth_model)
        except ValueError:
            return
        position = message.pose[index].position
        orientation = message.pose[index].orientation
        yaw = math.atan2(
            2.0 * (orientation.w * orientation.z
                   + orientation.x * orientation.y),
            1.0 - 2.0 * (orientation.y * orientation.y
                         + orientation.z * orientation.z))
        self._local_origin = [position.x, position.y, yaw]

    def _on_shared_origin(self, message):
        """Inherit the camera_init anchor resolved by the F1 runner."""
        if self._shared_origin is not None:
            return
        try:
            values = json.loads(message.data)
            origin = [float(v) for v in values]
        except (TypeError, ValueError):
            return
        if len(origin) == 3:
            self._shared_origin = origin

    def _on_locomotion_ready(self, message):
        with self._lock:
            self._locomotion_ready = bool(message.data)

    def _on_execution_result(self, message):
        try:
            candidate = json.loads(message.data)
        except (TypeError, ValueError):
            return
        with self._lock:
            self._execution_results.append(candidate)


def main():
    rospy.init_node("waypoint_tour_manager")
    try:
        waypoints = WaypointTourRunner._normalize_waypoints(
            rospy.get_param("~waypoints", ""))
    except ValueError as error:
        rospy.logfatal("[TOUR] invalid waypoints: %s", error)
        rospy.signal_shutdown("waypoint_tour_invalid_config")
        return
    runner = WaypointTourRunner(
        floor_number=int(rospy.get_param("~floor_number", 1)),
        output_dir=str(rospy.get_param("~output_dir", "")),
        waypoints=waypoints,
        odom_topic=str(rospy.get_param("~odom_topic", "/Odometry")),
        default_speed=float(rospy.get_param("~default_waypoint_speed", 1.00)),
        default_lateral_speed=float(rospy.get_param(
            "~default_lateral_speed_limit", 0.35)),
        waypoint_timeout=float(rospy.get_param("~waypoint_timeout_sec", 60.0)),
        startup_timeout=float(rospy.get_param("~startup_timeout", 120.0)),
        waypoint_frame=str(rospy.get_param("~waypoint_frame", "world")),
        ground_truth_topic=str(rospy.get_param(
            "~ground_truth_topic", "/gazebo/model_states")),
        ground_truth_model=str(rospy.get_param(
            "~ground_truth_model", "a1_gazebo")),
        publish_baseline_state=True)
    if not runner.wait_for_startup():
        runner.termination_reason = "STARTUP_TIMEOUT"
        runner._write_summary()
        rospy.logerr("[TOUR] startup gate not met within %.0f s "
                     "(odom or locomotion_ready missing)",
                     runner._startup_timeout)
        rospy.signal_shutdown("waypoint_tour_startup_timeout")
        return
    runner.publish_state("MISSION_CLOCK_START", startup_complete=True)
    runner.publish_state("WAYPOINT_TOUR_START")
    rospy.loginfo("[TOUR] startup ready; executing %d waypoints",
                  len(runner.waypoints))
    if runner.run():
        # The stair manager latches on the substring "STAIR_WAIT_ZONE" and
        # drives its own truth-guided approach from here on.
        runner.publish_state("STAIR_WAIT_ZONE")
        rospy.loginfo("[TOUR] tour complete; STAIR_WAIT_ZONE published, "
                      "stair manager takes over")
        rospy.spin()
    else:
        runner.publish_state("WAYPOINT_TOUR_FAILED",
                             reason=runner.termination_reason)
        rospy.logerr("[TOUR] failed: %s", runner.termination_reason)
        rospy.signal_shutdown("waypoint_tour_failed")


if __name__ == "__main__":
    main()
