#!/usr/bin/env python3
"""Finish the three-floor RL mission at the first-floor lobby.

The production SimEnv floor and stair managers own the mission through the
F3 -> F2 -> F1 descent.  This node observes all policy hand-offs, takes over
only after ``FIRST_FLOOR_RETURNED``, reloads the plane policy, and walks the
physical A1 out of the stairwell to the configured lobby point.  Motion is
always executed by the Unitree RL joint controller; this node publishes only
body-frame ``Twist`` commands and never calls Gazebo set-model-state services.
"""

import json
import math
import os
import threading
import time

import rospy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, String


def clamp(value, lower, upper):
    return max(float(lower), min(float(upper), float(value)))


def normalize_angle(angle):
    return math.atan2(math.sin(float(angle)), math.cos(float(angle)))


def policy_kind(value):
    """Classify a policy path/status as ``plane``, ``stair`` or ``unknown``."""
    text = str(value).lower()
    if "stair" in text:
        return "stair"
    if "plane" in text or "flat" in text:
        return "plane"
    return "unknown"


def policy_sequence_satisfied(events):
    """Return whether entrance and building-stair gait switches were observed."""
    kinds = []
    for event in events:
        kind = policy_kind(event.get("policy", event) if isinstance(event, dict) else event)
        if kind != "unknown" and (not kinds or kinds[-1] != kind):
            kinds.append(kind)
    required = [
        "stair", "plane",  # main-entrance step then lobby flat
        "stair", "plane",  # F1 -> F2
        "stair", "plane",  # F2 -> F3
        "stair", "plane",  # F3 descent then returned lobby
    ]
    cursor = 0
    for kind in kinds:
        if cursor < len(required) and kind == required[cursor]:
            cursor += 1
    return cursor == len(required)


def inside_bounds(x, y, bounds, margin=0.0):
    margin = max(0.0, float(margin))
    return (
        float(bounds["x_min"]) + margin <= float(x) <= float(bounds["x_max"]) - margin
        and float(bounds["y_min"]) + margin <= float(y) <= float(bounds["y_max"]) - margin
    )


def command_for_lobby_waypoint(pose, target, maximum_speed, maximum_yaw_rate,
                               position_tolerance, heading_release_tolerance,
                               target_yaw=None, target_heading_tolerance=0.18):
    """Compute a turn-before-walk command for the plane RL policy.

    ``pose`` is ``(x, y, z, yaw)`` in the world frame.  The returned linear
    command is body-forward; the robot first aligns its body with the target,
    which avoids asking the learned policy for a large simultaneous lateral
    and yaw motion at the stair exit.
    """
    dx = float(target[0]) - float(pose[0])
    dy = float(target[1]) - float(pose[1])
    distance = math.hypot(dx, dy)
    if distance <= float(position_tolerance):
        if target_yaw is None:
            return 0.0, 0.0, 0.0, distance, 0.0, True
        heading_error = normalize_angle(float(target_yaw) - float(pose[3]))
        if abs(heading_error) <= float(target_heading_tolerance):
            return 0.0, 0.0, 0.0, distance, heading_error, True
        yaw_rate = clamp(
            1.2 * heading_error, -maximum_yaw_rate, maximum_yaw_rate)
        return 0.0, 0.0, yaw_rate, distance, heading_error, False

    desired_yaw = math.atan2(dy, dx)
    heading_error = normalize_angle(desired_yaw - float(pose[3]))
    yaw_rate = clamp(1.2 * heading_error, -maximum_yaw_rate, maximum_yaw_rate)

    if abs(heading_error) > float(heading_release_tolerance):
        forward = 0.0
    else:
        forward = min(float(maximum_speed), max(0.18, 0.70 * distance))
        forward *= max(0.30, math.cos(heading_error))
    return forward, 0.0, yaw_rate, distance, heading_error, False


def limit_lobby_forward_acceleration(previous_speed, requested_speed, delta_sec,
                                     maximum_acceleration,
                                     initial_forward_speed=0.18):
    """Shape the plane-policy forward command after an in-place turn.

    A zero request is applied immediately so the turn-before-walk safety gate
    remains authoritative.  A positive request starts at the policy's benign
    low-speed command and can then increase only at ``maximum_acceleration``.
    Requested slow-downs are also applied immediately; the limiter therefore
    cannot make the robot overshoot a waypoint or keep walking after the
    heading gate closes.
    """
    previous = max(0.0, float(previous_speed))
    requested = max(0.0, float(requested_speed))
    if requested <= 0.0:
        return 0.0
    if requested <= previous:
        return requested
    if previous <= 0.0:
        return min(requested, max(0.0, float(initial_forward_speed)))
    increase = max(0.0, float(maximum_acceleration)) * max(0.0, float(delta_sec))
    return min(requested, previous + increase)


def upright_z_from_quaternion(q):
    """World-up component of the robot body's local z axis."""
    return 1.0 - 2.0 * (float(q.x) * float(q.x) + float(q.y) * float(q.y))


def attitude_loss_debounce(previous_since, upright_z, minimum_upright_z,
                           now, minimum_duration):
    if float(upright_z) >= float(minimum_upright_z):
        return None, False
    since = float(now) if previous_since is None else float(previous_since)
    return since, float(now) - since >= float(minimum_duration)


def decode_route_state(text):
    """Decode a sequencer state while retaining plain-string compatibility."""
    try:
        payload = json.loads(str(text))
    except (TypeError, ValueError):
        payload = {"phase": str(text)}
    return payload if isinstance(payload, dict) else {"phase": str(text)}


def task_stage_for_route_state(payload):
    """Map sequencer phases to the eight user-visible timed stages."""
    phase = str(payload.get("phase", "")).upper()
    floor = int(payload.get("floor", 0) or 0)
    if phase == "ENTER_MAIN_ENTRANCE_STAIR_RL":
        return "entrance_step_stair_rl"
    if phase == "EXPLORE_FLOOR" and floor in (1, 2, 3):
        return "floor_{}_exploration".format(floor)
    if phase in ("PRELOAD_STAIR_RL_FOR_TRANSITION", "RELEASE_TO_STAIR_RL"):
        return {
            1: "stair_f1_to_f2",
            2: "stair_f2_to_f3",
            3: "stair_f3_to_f1_descent",
        }.get(floor)
    return None


def duration_within_budget(duration_sec, budget_sec):
    try:
        return 0.0 <= float(duration_sec) <= float(budget_sec)
    except (TypeError, ValueError):
        return False


def load_config(path):
    with open(path, "r", encoding="utf-8") as stream:
        config = json.load(stream)
    if config.get("schema") != "scanplanner_three_floor_rl_mission_v1":
        raise ValueError("unsupported mission config schema")
    section = config.get("return_to_lobby", {})
    if not section.get("waypoints"):
        raise ValueError("return_to_lobby.waypoints must not be empty")
    return config


class ThreeFloorRLMissionSupervisor:
    def __init__(self):
        self.config_path = rospy.get_param(
            "~mission_config",
            "/workspace/SimEnv/src/simenv_bridge/config/three_floor_rl_mission.json",
        )
        self.output_dir = rospy.get_param(
            "~output_dir", "/workspace/SimEnv/results/scanplanner_three_floor_rl"
        )
        self.model_name = rospy.get_param("~model_name", "a1_gazebo")
        self.rate_hz = float(rospy.get_param("~rate", 20.0))
        self.mission_timeout = float(rospy.get_param("~mission_timeout_sec", 2400.0))
        self.progress_timeout = float(rospy.get_param("~progress_timeout_sec", 15.0))
        self.minimum_upright_z = float(rospy.get_param("~minimum_upright_z", 0.55))
        self.minimum_upright_duration = max(0.0, float(rospy.get_param(
            "~minimum_upright_duration_sec", 0.60)))

        self.config = load_config(self.config_path)
        self.runtime = self.config["runtime"]
        self.return_config = self.config["return_to_lobby"]
        self.spawn_config = self.config.get("spawn_pose", {})
        self.timing_config = self.config.get("task_timing", {})
        self.task_budget = float(
            self.timing_config.get("maximum_exploration_duration_sec", 600.0))
        self.plane_policy = str(self.runtime["plane_policy"])
        self.trigger_token = str(self.return_config["trigger_token"])
        self.maximum_speed = float(self.return_config["maximum_speed"])
        self.maximum_yaw_rate = float(self.return_config["maximum_yaw_rate"])
        self.position_tolerance = float(self.return_config["position_tolerance"])
        self.heading_release_tolerance = float(
            self.return_config["heading_release_tolerance"]
        )
        self.waypoint_timeout = float(self.return_config["waypoint_timeout_sec"])
        self.policy_timeout = float(self.return_config["policy_timeout_sec"])
        self.plane_fast_takeover = bool(rospy.get_param(
            "~plane_fast_takeover", True))
        self.plane_fast_blend = max(0.25, float(rospy.get_param(
            "~plane_fast_takeover_blend_seconds", 1.25)))
        self.plane_fast_zero_hold = max(0.10, float(rospy.get_param(
            "~plane_fast_takeover_zero_hold_seconds", 0.75)))
        self.lobby_heading_stable = max(0.0, float(rospy.get_param(
            "~lobby_heading_stable_sec",
            self.return_config.get("heading_stable_sec", 0.30))))
        self.lobby_linear_acceleration = max(0.05, float(rospy.get_param(
            "~lobby_linear_acceleration_mps2",
            self.return_config.get("linear_acceleration_mps2", 0.55))))
        self.lobby_initial_forward_speed = max(0.0, float(rospy.get_param(
            "~lobby_initial_forward_speed_mps",
            self.return_config.get("initial_forward_speed_mps", 0.18))))

        os.makedirs(self.output_dir, exist_ok=True)
        self.summary_path = os.path.join(
            self.output_dir, "three_floor_rl_mission_summary.json"
        )
        self.timing_path = os.path.join(
            self.output_dir,
            str(self.timing_config.get("output_file", "mission_stage_timing.json")),
        )

        self._lock = threading.RLock()
        self._write_lock = threading.Lock()
        self._pose = None
        self._upright_z = None
        self._triggered = False
        self._locomotion_ready = False
        self._ready_monotonic = None
        self._plane_ack_monotonic = None
        self._policy_events = []
        self._mission_events = []
        self._phase = "WAIT_THREE_FLOOR_DESCENT"
        self._failure = None
        self._completed = False
        self._started_wall_time = time.time()
        self._started_monotonic = time.monotonic()
        self._exploration_started_wall_time = None
        self._exploration_started_monotonic = None
        self._exploration_started_sim_elapsed = None
        self._exploration_start_pose = None
        self._exploration_start_requested = False
        self._task_stage = None
        self._task_stage_started_monotonic = None
        self._task_stage_started_sim_elapsed = None
        self._task_stages = []
        self._sim_clock_last = None
        self._sim_clock_elapsed = 0.0
        self._sim_clock_resets = 0
        self._return_started_wall_time = None
        self._visited_return_waypoints = []
        self._attitude_low_since = None

        self.command_pub = rospy.Publisher(
            self.return_config["command_topic"], Twist, queue_size=2
        )
        self.pause_pub = rospy.Publisher(
            self.return_config["goal_executor_pause_topic"],
            Bool,
            queue_size=1,
            latch=True,
        )
        self.policy_pub = rospy.Publisher(
            "/simenv/rl_policy_request", String, queue_size=1, latch=True
        )
        self.state_pub = rospy.Publisher(
            "/scanplanner/three_floor_mission_state", String, queue_size=1, latch=True
        )
        self.complete_pub = rospy.Publisher(
            "/scanplanner/three_floor_mission_complete", Bool, queue_size=1, latch=True
        )
        self.abort_pub = rospy.Publisher(
            "/simenv/mission_abort", Bool, queue_size=1, latch=True
        )
        self.exploration_active_pub = rospy.Publisher(
            "/simenv/exploration_active", Bool, queue_size=1, latch=True
        )
        self.exploration_clock_ready_pub = rospy.Publisher(
            "/simenv/exploration_clock_ready", Bool, queue_size=1, latch=True
        )
        self.exploration_clock_ready_pub.publish(Bool(data=False))

        rospy.Subscriber(
            self.return_config["trigger_topic"], String, self._on_return_trigger, queue_size=2
        )
        rospy.Subscriber(
            "/simenv/exploration_active", Bool,
            self._on_exploration_active, queue_size=5)
        rospy.Subscriber("/rl_takeover_status", String, self._on_policy_status, queue_size=20)
        rospy.Subscriber("/locomotion_ready", Bool, self._on_locomotion_ready, queue_size=5)
        rospy.Subscriber("/gazebo/model_states", ModelStates, self._on_model_states, queue_size=5)
        for topic in (
            "/simenv/baseline_state",
            "/simenv/stair_transition_state",
            "/simenv/second_floor_state",
            "/simenv/second_to_third_floor_stair_state",
            "/simenv/third_floor_state",
            "/simenv/third_to_first_floor_stair_state",
            "/scanplanner/route_state",
        ):
            rospy.Subscriber(topic, String, self._on_mission_state, callback_args=topic, queue_size=10)

        self.complete_pub.publish(Bool(data=False))
        self.abort_pub.publish(Bool(data=False))
        self._record_event("supervisor_started", config=self.config_path)
        self._write_summary("running")
        self._write_timing("preparing")
        rospy.on_shutdown(self._on_shutdown)

    def _record_event(self, event, **fields):
        item = {"event": str(event), "wall_time": round(time.time(), 3)}
        item.update(fields)
        with self._lock:
            self._mission_events.append(item)
            self._mission_events = self._mission_events[-500:]

    def _sample_sim_clock(self):
        """Accumulate positive ROS-time deltas across Gazebo clock resets."""
        try:
            current = float(rospy.get_time())
        except (AttributeError, TypeError, ValueError):
            return
        if not math.isfinite(current):
            return
        with self._lock:
            previous = self._sim_clock_last
            if previous is not None:
                delta = current - previous
                if delta >= 0.0:
                    self._sim_clock_elapsed += delta
                else:
                    self._sim_clock_resets += 1
            self._sim_clock_last = current

    def _task_elapsed(self):
        with self._lock:
            started = self._exploration_started_sim_elapsed
            current = self._sim_clock_elapsed
        return None if started is None else max(0.0, current - started)

    def _start_exploration_clock(self):
        started_now = False
        with self._lock:
            if self._exploration_started_monotonic is not None:
                clock_ready = True
            elif self._pose is None:
                self._exploration_start_requested = True
                clock_ready = False
            else:
                self._exploration_started_wall_time = time.time()
                self._exploration_started_monotonic = time.monotonic()
                self._exploration_started_sim_elapsed = self._sim_clock_elapsed
                self._exploration_start_pose = list(self._pose)
                self._exploration_start_requested = False
                started_now = True
                clock_ready = True
        if not clock_ready:
            return
        self.exploration_clock_ready_pub.publish(Bool(data=True))
        if not started_now:
            return
        self.exploration_active_pub.publish(Bool(data=True))
        self._record_event(
            "exploration_clock_started",
            preparation_excluded=True,
            budget_sec=self.task_budget,
            time_basis="ros_simulation_time",
            spawn_truth_pose=self._exploration_start_pose,
        )
        self._write_timing("running")

    def _on_exploration_active(self, message):
        if bool(message.data):
            self._start_exploration_clock()

    def _transition_task_stage(self, stage, source):
        if not stage:
            return
        self._start_exploration_clock()
        now = time.monotonic()
        with self._lock:
            if self._task_stage == stage:
                return
            if self._task_stage is not None:
                start = self._task_stage_started_monotonic
                start_sim = self._task_stage_started_sim_elapsed
                sim_start_elapsed = (
                    start_sim - self._exploration_started_sim_elapsed)
                sim_end_elapsed = (
                    self._sim_clock_elapsed -
                    self._exploration_started_sim_elapsed)
                self._task_stages.append({
                    "name": self._task_stage,
                    "start_elapsed_sec": round(sim_start_elapsed, 3),
                    "end_elapsed_sec": round(sim_end_elapsed, 3),
                    "duration_sec": round(sim_end_elapsed - sim_start_elapsed, 3),
                    "sim_duration_sec": round(sim_end_elapsed - sim_start_elapsed, 3),
                    "wall_duration_sec": round(now - start, 3),
                    "end_reason": str(source),
                })
            self._task_stage = str(stage)
            self._task_stage_started_monotonic = now
            self._task_stage_started_sim_elapsed = self._sim_clock_elapsed
        self._record_event("task_stage", stage=stage, source=source)
        self._write_timing("running")

    def _close_task_clock(self, status):
        now = time.monotonic()
        with self._lock:
            if (self._exploration_started_monotonic is not None and
                    self._task_stage is not None):
                sim_start_elapsed = (
                    self._task_stage_started_sim_elapsed -
                    self._exploration_started_sim_elapsed)
                sim_end_elapsed = (
                    self._sim_clock_elapsed -
                    self._exploration_started_sim_elapsed)
                self._task_stages.append({
                    "name": self._task_stage,
                    "start_elapsed_sec": round(sim_start_elapsed, 3),
                    "end_elapsed_sec": round(sim_end_elapsed, 3),
                    "duration_sec": round(sim_end_elapsed - sim_start_elapsed, 3),
                    "sim_duration_sec": round(sim_end_elapsed - sim_start_elapsed, 3),
                    "wall_duration_sec": round(
                        now - self._task_stage_started_monotonic, 3),
                    "end_reason": str(status),
                })
                self._task_stage = None
                self._task_stage_started_monotonic = None
                self._task_stage_started_sim_elapsed = None
        self.exploration_active_pub.publish(Bool(data=False))
        self._write_timing(status)

    def _check_task_deadline(self):
        elapsed = self._task_elapsed()
        if elapsed is None or elapsed < self.task_budget:
            return True
        with self._lock:
            if self._failure is None:
                self._failure = "exploration_deadline_exceeded_{:.1f}s".format(
                    self.task_budget)
        self.abort_pub.publish(Bool(data=True))
        self._write_timing("failed")
        return False

    def _write_timing(self, status):
        now_mono = time.monotonic()
        with self._lock:
            started = self._exploration_started_monotonic
            wall_duration = (
                None if started is None else max(0.0, now_mono - started))
            sim_duration = (
                None if self._exploration_started_sim_elapsed is None else
                max(0.0, self._sim_clock_elapsed -
                    self._exploration_started_sim_elapsed)
            )
            current_stage = None
            if self._task_stage is not None:
                sim_stage_start = (
                    self._task_stage_started_sim_elapsed -
                    self._exploration_started_sim_elapsed)
                current_stage = {
                    "name": self._task_stage,
                    "start_elapsed_sec": round(sim_stage_start, 3),
                    "elapsed_sec": round(
                        self._sim_clock_elapsed -
                        self._task_stage_started_sim_elapsed, 3),
                    "wall_elapsed_sec": round(
                        now_mono - self._task_stage_started_monotonic, 3),
                }
            payload = {
                "schema": "scanplanner_mission_stage_timing_v1",
                "status": str(status),
                "preparation_excluded": True,
                "clock_definition": (
                    "ROS/Gazebo simulation time from EXPLORATION_STARTED; algorithm "
                    "preparation and sensor/policy warm-up are excluded"
                ),
                "time_basis": "ros_simulation_time",
                "maximum_duration_sec": self.task_budget,
                "budget_met": (
                    duration_within_budget(sim_duration, self.task_budget)
                    if sim_duration is not None and status in ("completed", "failed")
                    else None
                ),
                "algorithm_preparation_duration_sec": round(
                    (self._exploration_started_monotonic or now_mono) -
                    self._started_monotonic, 3),
                "exploration_started_wall_time": (
                    round(self._exploration_started_wall_time, 3)
                    if self._exploration_started_wall_time is not None else None
                ),
                "exploration_finished_wall_time": (
                    round(time.time(), 3)
                    if status in ("completed", "failed", "interrupted") else None
                ),
                "total_duration_sec": (
                    round(sim_duration, 3) if sim_duration is not None else None
                ),
                "sim_duration_sec": (
                    round(sim_duration, 3) if sim_duration is not None else None
                ),
                "wall_duration_sec": (
                    round(wall_duration, 3) if wall_duration is not None else None
                ),
                "sim_clock_reset_count": self._sim_clock_resets,
                "required_stages": list(
                    self.timing_config.get("required_stages", [])),
                "stages": list(self._task_stages),
                "current_stage": current_stage,
                "spawn_truth_pose_at_start": self._exploration_start_pose,
                "failure_reason": self._failure,
            }
        temporary = self.timing_path + ".tmp"
        with self._write_lock:
            try:
                with open(temporary, "w", encoding="utf-8") as stream:
                    json.dump(payload, stream, ensure_ascii=False, indent=2,
                              sort_keys=True)
                    stream.write("\n")
                os.replace(temporary, self.timing_path)
            except OSError as error:
                rospy.logwarn_throttle(
                    10.0, "Cannot write mission stage timing: %s", error)

    def _set_phase(self, phase, **fields):
        with self._lock:
            self._phase = str(phase)
        payload = {"phase": str(phase), "wall_time": round(time.time(), 3)}
        payload.update(fields)
        self.state_pub.publish(String(data=json.dumps(payload, sort_keys=True)))
        self._record_event("phase", **payload)
        self._write_summary("running")

    def _on_return_trigger(self, message):
        if str(message.data).strip() != self.trigger_token:
            return
        with self._lock:
            if self._triggered:
                return
            self._triggered = True
            self._return_started_wall_time = time.time()
        self._record_event("first_floor_descent_complete", token=self.trigger_token)
        self._transition_task_stage(
            "return_to_first_floor_lobby", "FIRST_FLOOR_RETURNED")

    def _on_policy_status(self, message):
        text = str(message.data)
        if "policy_reloaded:" not in text and "policy_reload_failed:" not in text:
            return
        kind = policy_kind(text)
        item = {
            "policy": kind,
            "status": text,
            "wall_time": round(time.time(), 3),
        }
        with self._lock:
            self._policy_events.append(item)
            if "policy_reloaded:" in text and kind == "plane":
                self._plane_ack_monotonic = time.monotonic()
            if "policy_reload_failed:" in text:
                self._failure = "policy_reload_failed:{}".format(text)
        self._record_event("policy_status", policy=kind, status=text)
        self._write_summary("running")

    def _on_locomotion_ready(self, message):
        with self._lock:
            self._locomotion_ready = bool(message.data)
            if message.data:
                self._ready_monotonic = time.monotonic()

    def _on_model_states(self, message):
        self._sample_sim_clock()
        try:
            index = message.name.index(self.model_name)
            pose = message.pose[index]
        except (ValueError, IndexError):
            return
        q = pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        with self._lock:
            self._pose = (
                float(pose.position.x),
                float(pose.position.y),
                float(pose.position.z),
                float(yaw),
            )
            self._upright_z = upright_z_from_quaternion(q)
            start_requested = self._exploration_start_requested
        if start_requested:
            self._start_exploration_clock()

    def _on_mission_state(self, message, topic):
        text = str(message.data)
        self._record_event("mission_state", topic=topic, state=text)
        route_phase = None
        if topic == "/scanplanner/route_state":
            payload = decode_route_state(text)
            phase = str(payload.get("phase", "")).upper()
            route_phase = phase
            if phase == "EXPLORATION_STARTED":
                self._start_exploration_clock()
            self._transition_task_stage(
                task_stage_for_route_state(payload), phase)
        upper = text.upper()
        failure_tokens = (
            "FAILED", "TIMEOUT", "FALL_DETECTED", "ALIGNMENT_LOST",
            "ATTITUDE_LOST",
        )
        # Structured route states can legitimately carry a diagnostic reason
        # containing the word "failed" while performing the one bounded local
        # rescan.  For that topic, only the phase is authoritative; otherwise
        # the recovery notification races the sequencer and aborts a retry that
        # subsequently passes.
        if route_phase is not None:
            upstream_failed = (
                route_phase.endswith("_FAILED") or
                route_phase in ("FAILED", "ABORTED", "INTERRUPTED"))
        else:
            upstream_failed = any(token in upper for token in failure_tokens)
        if upstream_failed:
            with self._lock:
                if not self._triggered:
                    self._failure = "upstream_failure:{}:{}".format(topic, text)

    def _zero(self):
        self.command_pub.publish(Twist())

    def _wait_for_trigger(self):
        rate = rospy.Rate(2.0)
        deadline = time.monotonic() + self.mission_timeout
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            if not self._check_task_deadline():
                return False
            with self._lock:
                if self._failure:
                    return False
                if self._triggered:
                    return True
            rate.sleep()
        with self._lock:
            self._failure = "mission_timeout_waiting_for_first_floor_return"
        return False

    def _restore_plane_policy(self):
        self._set_phase("RESTORE_PLANE_RL_POLICY", policy=self.plane_policy)
        self.pause_pub.publish(Bool(data=True))
        self._zero()
        # Discard the plane-policy acknowledgement from the earlier F2/F3
        # hand-offs.  The descent ends with the stair policy active, so only a
        # fresh acknowledgement and a fresh ready edge after this request are
        # valid for the final lobby walk.
        with self._lock:
            self._plane_ack_monotonic = None
            self._ready_monotonic = None
        request_started = time.monotonic()
        if self.plane_fast_takeover:
            rospy.set_param(
                "/simenv/plane_fast_takeover_blend_seconds",
                self.plane_fast_blend)
            rospy.set_param(
                "/simenv/plane_fast_takeover_zero_hold_seconds",
                self.plane_fast_zero_hold)
            # Publish the one-shot enable flag last; State_RL consumes it
            # only for a requested plane-policy hot reload.
            rospy.set_param("/simenv/plane_fast_takeover_enabled", True)
        deadline = time.monotonic() + self.policy_timeout
        next_request = 0.0
        rate = rospy.Rate(10.0)
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            if not self._check_task_deadline():
                return False
            now = time.monotonic()
            if now >= next_request:
                self.policy_pub.publish(String(data=self.plane_policy))
                next_request = now + 0.75
            with self._lock:
                ack = self._plane_ack_monotonic
                ready_at = self._ready_monotonic
                failure = self._failure
                ready = self._locomotion_ready
            if failure:
                return False
            # Reload status and locomotion readiness are emitted from
            # different controller callbacks.  Require both to be fresh for
            # this request without assuming which callback is delivered first.
            if (ack is not None and ack >= request_started and ready and
                    ready_at is not None and ready_at >= request_started):
                self._record_event("plane_policy_ready", policy=self.plane_policy)
                return True
            self._zero()
            rate.sleep()
        with self._lock:
            self._failure = "plane_policy_restore_timeout"
        return False

    def _drive_waypoint(self, waypoint, index):
        target = (float(waypoint["x"]), float(waypoint["y"]))
        deadline = time.monotonic() + self.waypoint_timeout
        progress_deadline = time.monotonic() + self.progress_timeout
        best_distance = float("inf")
        previous_forward = 0.0
        previous_command_time = time.monotonic()
        heading_stable_since = None
        ramp_announced = False
        self._record_event(
            "return_waypoint_started", index=int(index),
            note=waypoint.get("note"), target=[target[0], target[1]])
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            if not self._check_task_deadline():
                return False
            now = time.monotonic()
            command_delta = min(0.25, max(0.0, now - previous_command_time))
            previous_command_time = now
            with self._lock:
                pose = self._pose
                upright_z = self._upright_z
                failure = self._failure
                ready = self._locomotion_ready
            if failure:
                return False
            if pose is None:
                previous_forward = 0.0
                heading_stable_since = None
                self._zero()
                rate.sleep()
                continue
            if upright_z is not None:
                self._attitude_low_since, attitude_lost = (
                    attitude_loss_debounce(
                        self._attitude_low_since, upright_z,
                        self.minimum_upright_z, now,
                        self.minimum_upright_duration))
                if self._attitude_low_since is not None:
                    previous_forward = 0.0
                    heading_stable_since = None
                    self._zero()
                    if attitude_lost:
                        with self._lock:
                            self._failure = "lobby_return_attitude_lost"
                        return False
                    rate.sleep()
                    continue
            if not ready:
                previous_forward = 0.0
                heading_stable_since = None
                self._zero()
                rate.sleep()
                continue

            requested_vx, vy, wz, distance, heading_error, reached = command_for_lobby_waypoint(
                pose,
                target,
                self.maximum_speed,
                self.maximum_yaw_rate,
                self.position_tolerance,
                self.heading_release_tolerance,
                target_yaw=waypoint.get("yaw"),
                target_heading_tolerance=float(waypoint.get(
                    "heading_tolerance", 0.18)),
            )
            if reached:
                self._zero()
                record = {
                    "index": int(index),
                    "note": waypoint.get("note"),
                    "target": [target[0], target[1]],
                    "reached_pose": [round(value, 4) for value in pose],
                    "distance_error_m": round(distance, 4),
                    "wall_time": round(time.time(), 3),
                }
                with self._lock:
                    self._visited_return_waypoints.append(record)
                self._record_event("return_waypoint_reached", **record)
                self._write_summary("running")
                rospy.sleep(0.15)
                return True

            if requested_vx > 0.0:
                if heading_stable_since is None:
                    heading_stable_since = now
                heading_stable_elapsed = max(0.0, now - heading_stable_since)
            else:
                heading_stable_since = None
                heading_stable_elapsed = 0.0

            ramp_request = (
                requested_vx
                if heading_stable_elapsed >= self.lobby_heading_stable else 0.0
            )
            vx = limit_lobby_forward_acceleration(
                previous_forward, ramp_request, command_delta,
                self.lobby_linear_acceleration,
                self.lobby_initial_forward_speed)
            previous_forward = vx
            if vx > 0.0 and not ramp_announced:
                ramp_announced = True
                self._record_event(
                    "lobby_forward_ramp_started", index=int(index),
                    requested_speed_mps=round(requested_vx, 3),
                    initial_speed_mps=round(vx, 3),
                    heading_stable_sec=round(heading_stable_elapsed, 3),
                    maximum_acceleration_mps2=self.lobby_linear_acceleration)

            if distance < best_distance - 0.05:
                best_distance = distance
                progress_deadline = time.monotonic() + self.progress_timeout
            elif time.monotonic() >= progress_deadline:
                with self._lock:
                    self._failure = "lobby_return_no_progress_at_waypoint_{}".format(index)
                return False

            command = Twist()
            command.linear.x = vx
            command.linear.y = vy
            command.angular.z = wz
            self.command_pub.publish(command)
            self.state_pub.publish(String(data=json.dumps({
                "phase": "RETURN_TO_FIRST_FLOOR_LOBBY",
                "waypoint_index": int(index),
                "waypoint_note": waypoint.get("note"),
                "distance_m": round(distance, 3),
                "heading_error_rad": round(heading_error, 3),
                "requested_forward_mps": round(requested_vx, 3),
                "command_forward_mps": round(vx, 3),
                "heading_stable_sec": round(heading_stable_elapsed, 3),
            }, sort_keys=True)))
            rate.sleep()
        with self._lock:
            self._failure = "lobby_return_waypoint_{}_timeout".format(index)
        return False

    def _finish(self):
        self._zero()
        rospy.sleep(0.25)
        with self._lock:
            pose = self._pose
        if pose is None or not inside_bounds(
            pose[0], pose[1], self.return_config["lobby_bounds"], margin=0.10
        ):
            with self._lock:
                self._failure = "final_pose_outside_first_floor_lobby"
            return False
        self.pause_pub.publish(Bool(data=False))
        self._completed = True
        self._phase = "MISSION_COMPLETE_AT_FIRST_FLOOR_LOBBY"
        self.complete_pub.publish(Bool(data=True))
        self.state_pub.publish(String(data=self._phase))
        self._record_event("mission_complete", final_pose=list(pose))
        self._close_task_clock("completed")
        self._write_summary("completed")
        rospy.loginfo(
            "Three-floor RL mission complete in first-floor lobby at (%.2f, %.2f, %.2f)",
            pose[0], pose[1], pose[2],
        )
        return True

    def _fail(self):
        self._zero()
        self.pause_pub.publish(Bool(data=True))
        with self._lock:
            reason = self._failure or "unknown_failure"
            self._phase = "MISSION_FAILED"
        self.complete_pub.publish(Bool(data=False))
        self.abort_pub.publish(Bool(data=True))
        self.state_pub.publish(String(data="MISSION_FAILED:" + reason))
        self._record_event("mission_failed", reason=reason)
        self._close_task_clock("failed")
        self._write_summary("failed")
        rospy.logerr("Three-floor RL mission failed: %s", reason)

    def _write_summary(self, status):
        with self._lock:
            exploration_sim_duration = (
                max(0.0, self._sim_clock_elapsed -
                    self._exploration_started_sim_elapsed)
                if self._exploration_started_sim_elapsed is not None else None)
            exploration_wall_duration = (
                time.monotonic() - self._exploration_started_monotonic
                if self._exploration_started_monotonic is not None else None)
            payload = {
                "schema": "scanplanner_three_floor_rl_summary_v1",
                "status": str(status),
                "phase": self._phase,
                "failure_reason": self._failure,
                "started_wall_time": round(self._started_wall_time, 3),
                "finished_wall_time": round(time.time(), 3),
                "duration_sec": round(time.monotonic() - self._started_monotonic, 3),
                "algorithm_preparation_excluded": True,
                "exploration_time_basis": "ros_simulation_time",
                "exploration_started_wall_time": (
                    round(self._exploration_started_wall_time, 3)
                    if self._exploration_started_wall_time is not None else None
                ),
                "exploration_duration_sec": (
                    round(exploration_sim_duration, 3)
                    if exploration_sim_duration is not None else None
                ),
                "wall_exploration_duration_sec": (
                    round(exploration_wall_duration, 3)
                    if exploration_wall_duration is not None else None
                ),
                "exploration_budget_sec": self.task_budget,
                "exploration_budget_met": (
                    duration_within_budget(
                        exploration_sim_duration,
                        self.task_budget,
                    ) if exploration_sim_duration is not None and
                    status in ("completed", "failed") else None
                ),
                "spawn_truth_pose_at_exploration_start": self._exploration_start_pose,
                "stage_timing_file": self.timing_path,
                "return_started_wall_time": (
                    round(self._return_started_wall_time, 3)
                    if self._return_started_wall_time is not None else None
                ),
                "plane_policy": self.plane_policy,
                "stair_policy": self.runtime.get("stair_policy"),
                "lobby_command_profile": {
                    "heading_stable_sec": self.lobby_heading_stable,
                    "linear_acceleration_mps2": self.lobby_linear_acceleration,
                    "initial_forward_speed_mps": self.lobby_initial_forward_speed,
                    "maximum_speed_mps": self.maximum_speed,
                },
                "policy_events": list(self._policy_events),
                "policy_sequence_satisfied": policy_sequence_satisfied(
                    self._policy_events
                ),
                "return_waypoints": list(self._visited_return_waypoints),
                "final_truth_pose": list(self._pose) if self._pose is not None else None,
                "lobby_bounds": dict(self.return_config["lobby_bounds"]),
                "mission_events": list(self._mission_events),
            }
        temporary = self.summary_path + ".tmp"
        with self._write_lock:
            try:
                with open(temporary, "w", encoding="utf-8") as stream:
                    json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
                    stream.write("\n")
                os.replace(temporary, self.summary_path)
            except OSError as error:
                rospy.logwarn_throttle(10.0, "Cannot write mission summary: %s", error)

    def _on_shutdown(self):
        self._zero()
        if not self._completed and self._failure is None:
            with self._lock:
                self._failure = "mission_interrupted"
            self._close_task_clock("interrupted")
            self._write_summary("interrupted")

    def run(self):
        self._set_phase("WAIT_THREE_FLOOR_DESCENT")
        if not self._wait_for_trigger():
            self._fail()
            return False
        if not self._restore_plane_policy():
            self._fail()
            return False
        self._set_phase("RETURN_TO_FIRST_FLOOR_LOBBY")
        for index, waypoint in enumerate(self.return_config["waypoints"]):
            if not self._drive_waypoint(waypoint, index):
                self._fail()
                return False
        if not self._finish():
            self._fail()
            return False
        return True


def main():
    rospy.init_node("three_floor_rl_mission_supervisor", anonymous=False)
    try:
        supervisor = ThreeFloorRLMissionSupervisor()
    except Exception as error:  # configuration errors are fatal and explicit
        rospy.logfatal("Cannot initialize three-floor RL supervisor: %s", error)
        raise
    supervisor.run()
    rospy.spin()


if __name__ == "__main__":
    main()
