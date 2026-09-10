#!/usr/bin/env python3
"""Floor-local fall detection and bounded upright recovery.

The stair managers own fall handling during a stair transition. This watchdog
uses explicit F1/F2/F3 bases and Gazebo truth only for fall safety; it never
changes normal FAST-LIO exploration.
"""

import math
import threading

import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, Joy
from std_msgs.msg import Bool, String


def yaw_from_quaternion(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def roll_pitch_from_quaternion(q):
    roll = math.atan2(
        2.0 * (q.w * q.x + q.y * q.z),
        1.0 - 2.0 * (q.x * q.x + q.y * q.y))
    sinp = 2.0 * (q.w * q.y - q.z * q.x)
    pitch = (math.copysign(math.pi / 2.0, sinp)
             if abs(sinp) >= 1.0 else math.asin(sinp))
    return roll, pitch


def quaternion_from_yaw(yaw):
    half = 0.5 * yaw
    return Quaternion(x=0.0, y=0.0, z=math.sin(half), w=math.cos(half))


def floor_base_from_state(topic, phase, floor_height, first_floor_base=0.0):
    """Return an explicit floor base, or None for an unrelated state."""
    definitions = (
        ("/simenv/second_floor_state", "SECOND_FLOOR_", 1),
        ("/simenv/third_floor_state", "THIRD_FLOOR_", 2),
    )
    phase = str(phase or "").strip()
    for expected_topic, prefix, floor_index in definitions:
        if str(topic) != expected_topic or not phase.startswith(prefix):
            continue
        if (phase.endswith("CORRIDOR_ENTRY_REACHED") or
                phase.endswith("EXPLORATION_START")):
            return (float(first_floor_base) +
                    floor_index * float(floor_height))
    return None


def stair_transition_is_active(phase):
    """Interpret the latched state of either stair manager."""
    phase = str(phase or "").strip()
    # All stair workers are resident for the complete round trip.  Their
    # pre-trigger WAIT states do not own locomotion or fall recovery.  In
    # particular the descent worker starts in WAIT_F3 while F1/F2/F3 room
    # exploration is running; treating that latched value as active disables
    # the flat-floor watchdog for the whole outbound mission.
    #
    # FIRST_FLOOR_HOME_FALL_RECOVERY_WAIT is also deliberately passive: the
    # descent manager enters it specifically to wait for this physical
    # recovery worker, so marking it active creates a circular wait.
    inactive_states = {
        "WAIT_F1",
        "WAIT_F2",
        "WAIT_F3",
        "FIRST_FLOOR_FINALIZING",
        "FIRST_FLOOR_HOME_FALL_RECOVERY_WAIT",
    }
    if not phase or phase in inactive_states:
        return False
    if (phase.endswith("HANDOFF_COMPLETE") or
            phase.endswith("HANDOFF_TIMEOUT")):
        return False
    # REACHED and SETTLE still retain stair-controller ownership.
    return True


def relative_fall_evidence(relative_z, roll, pitch, roll_limit, pitch_limit,
                           z_limit, flat_low_margin=0.10):
    """Detect side-lying and belly-down falls above one explicit floor."""
    tilt_bad = abs(roll) > roll_limit or abs(pitch) > pitch_limit
    flat_low = relative_z < float(z_limit) - float(flat_low_margin)
    return (tilt_bad and relative_z < float(z_limit)) or flat_low


def recovery_truth_sample_is_stable(
        z, expected_z, roll, pitch, linear_speed, angular_speed,
        height_tolerance, tilt_limit, linear_speed_limit,
        angular_speed_limit, relative_z, minimum_relative_z):
    """Test one fresh truth sample for safe recovery release."""
    values = (z, expected_z, roll, pitch, linear_speed, angular_speed)
    if not all(math.isfinite(float(value)) for value in values):
        return False
    return (
        float(relative_z) >= float(minimum_relative_z) and
        abs(float(z) - float(expected_z)) <= float(height_tolerance) and
        abs(float(roll)) <= float(tilt_limit) and
        abs(float(pitch)) <= float(tilt_limit) and
        float(linear_speed) <= float(linear_speed_limit) and
        float(angular_speed) <= float(angular_speed_limit))


class CompliantFallRecovery:
    _STAIR_TOPICS = (
        "/simenv/stair_transition_state",
        "/simenv/second_to_third_floor_stair_state",
        # The round-trip descent controller owns support-height and fall
        # handling while the robot is physically stepping down.  Without
        # this topic the flat-floor watchdog keeps the F3 floor base latched
        # at 5.20 m, interprets the first tread (truth z ~= 5.34 m) as a
        # belly-down fall, and teleports the robot back to z=5.75 m.  That
        # external reset then looks like a real flight fall to the descent
        # manager and tears down the required roslaunch process.
        "/simenv/third_to_first_floor_stair_state",
    )

    def __init__(self):
        self._lock = threading.RLock()
        self._pose = None
        self._imu = None
        self._truth = None
        self._truth_twist = None
        self._truth_received_at = None
        self._truth_sample_sequence = 0

        self._recovering = False
        self._recover_step = 0
        self._recover_step_time = rospy.Time.now()
        self._recovery_started_at = None
        self._recovery_floor_base = None
        self._recovery_expected_healthy_z = None
        self._recovery_release_streak = 0
        self._recovery_last_truth_sequence = -1
        self._last_recover = rospy.Time(0)

        self._cooldown = float(rospy.get_param("~recover_cooldown", 8.0))
        # z_limit and upright_z remain backward-compatible heights above base.
        self._upright_z = float(rospy.get_param("~upright_z", 0.55))
        self._standing_body_height = float(rospy.get_param(
            "~standing_body_height", 0.30))
        self._roll_limit = float(rospy.get_param("~roll_limit", 0.85))
        self._pitch_limit = float(rospy.get_param("~pitch_limit", 0.85))
        self._z_limit = float(rospy.get_param("~z_limit", 0.20))
        self._flat_low_margin = float(rospy.get_param(
            "~flat_low_margin", 0.10))
        self._start_grace = float(rospy.get_param(
            "~start_grace_seconds", 25.0))
        self._confirm_count = max(
            1, int(rospy.get_param("~confirm_count", 8)))
        self._candidate_clear_confirm_count = max(1, int(rospy.get_param(
            "~candidate_clear_confirm_count", 2)))
        self._use_gazebo_reset = bool(rospy.get_param(
            "~use_gazebo_reset", False))

        self._first_floor_base = float(rospy.get_param(
            "~first_floor_base_z", 0.0))
        self._floor_height = float(rospy.get_param("~floor_height", 2.60))
        self._floor_base = self._first_floor_base
        self._healthy_truth_z = None
        # Last upright, supported pose; do not reset at a sliding fallen pose.
        self._healthy_truth_anchor = None
        self._healthy_anchor_speed_limit = max(0.01, float(
            rospy.get_param("~healthy_anchor_speed_limit_mps", 0.12)))
        self._healthy_anchor_yaw_rate_limit = max(0.01, float(
            rospy.get_param("~healthy_anchor_yaw_rate_limit_rps", 0.20)))
        self._stair_active_by_topic = {
            topic: False for topic in self._STAIR_TOPICS}

        self._truth_maximum_age = max(0.05, float(rospy.get_param(
            "~truth_maximum_age_sec", 0.50)))
        self._recovery_height_tolerance = max(0.01, float(rospy.get_param(
            "~recovery_release_height_tolerance", 0.08)))
        self._minimum_healthy_relative_z = max(0.0, float(
            rospy.get_param("~minimum_healthy_relative_z", 0.27)))
        self._recovery_release_tilt = max(0.01, float(rospy.get_param(
            "~recovery_release_tilt", 0.35)))
        self._recovery_release_linear_speed = max(0.0, float(rospy.get_param(
            "~recovery_release_linear_speed", 0.12)))
        self._recovery_release_angular_speed = max(0.0, float(rospy.get_param(
            "~recovery_release_angular_speed", 0.25)))
        self._recovery_release_confirm_count = max(1, int(rospy.get_param(
            "~recovery_release_confirm_count", 5)))
        self._recovery_minimum_rl_seconds = max(0.0, float(rospy.get_param(
            "~recovery_minimum_rl_seconds", 2.0)))
        # Bound active reset attempts; timeout enters a fail-closed stuck state.
        requested_maximum = float(rospy.get_param(
            "~recovery_maximum_duration_sec", 12.0))
        self._recovery_maximum_duration = min(
            14.0, max(7.0, requested_maximum))
        self._recovery_retry_limit = max(0, min(2, int(rospy.get_param(
            "~recovery_retry_limit", 1))))
        self._recovery_retry_count = 0

        self._recovery_count = 0
        self._fall_streak = 0
        self._candidate_clear_streak = 0
        self._candidate_floor_base = None
        self._candidate_expected_healthy_z = None
        self._candidate_last_truth_sequence = -1
        self._recovery_stuck = False
        self._mapping_pause_asserted = False
        self._node_start = rospy.Time.now()

        self._hold_pub = rospy.Publisher(
            "/simenv/cmd_vel_hold", Bool, queue_size=1, latch=True)
        self._joy_pub = rospy.Publisher("/joy", Joy, queue_size=3)
        self._recovery_state_pub = rospy.Publisher(
            "/simenv/fall_recovery_active", Bool, queue_size=1, latch=True)
        self._status_pub = rospy.Publisher(
            "/simenv/fall_recovery_status", String, queue_size=1, latch=True)
        self._recovery_state_pub.publish(Bool(data=False))
        self._status_pub.publish(String(data="IDLE"))
        self._hold_pub.publish(Bool(data=False))
        rospy.Subscriber("/state_estimation", Odometry,
                         self._on_odom, queue_size=10)
        rospy.Subscriber("/trunk_imu", Imu, self._on_imu, queue_size=20)
        rospy.Subscriber("/simenv/second_floor_truth_odometry", Odometry,
                         self._on_truth_odometry, queue_size=5)
        for topic in self._STAIR_TOPICS:
            rospy.Subscriber(
                topic, String,
                lambda message, name=topic: self._on_stair_state(
                    name, message),
                queue_size=5)
        for topic in ("/simenv/second_floor_state",
                      "/simenv/third_floor_state"):
            rospy.Subscriber(
                topic, String,
                lambda message, name=topic: self._on_floor_state(
                    name, message),
                queue_size=5)
        rospy.Timer(rospy.Duration(0.1), self._on_timer)
        self._set_model_state = rospy.ServiceProxy(
            "/gazebo/set_model_state", SetModelState)
        rospy.loginfo(
            "Compliant fall recovery armed (floor_base=%.2f height=%.2f).",
            self._floor_base, self._floor_height)

    def _on_odom(self, message):
        with self._lock:
            self._pose = message

    def _on_imu(self, message):
        with self._lock:
            self._imu = message

    def _on_truth_odometry(self, message):
        with self._lock:
            self._truth = message.pose.pose
            self._truth_twist = message.twist.twist
            self._truth_received_at = rospy.Time.now()
            self._truth_sample_sequence += 1

    def _on_floor_state(self, topic, message):
        new_base = floor_base_from_state(
            topic, message.data, self._floor_height, self._first_floor_base)
        if new_base is None:
            return
        changed = False
        stair_topic = (
            "/simenv/stair_transition_state"
            if topic == "/simenv/second_floor_state"
            else "/simenv/second_to_third_floor_stair_state")
        with self._lock:
            # A REACHED/SETTLE state is latched by the stair manager. The
            # explicit flat-floor entry is the ownership handoff that clears
            # that latch for this watchdog.
            self._stair_active_by_topic[stair_topic] = False
            if abs(new_base - self._floor_base) > 1.0e-6:
                self._floor_base = new_base
                self._healthy_truth_z = None
                self._healthy_truth_anchor = None
                changed = True
        if changed:
            rospy.loginfo(
                "Fall recovery floor base locked by %s=%s: z=%.2f.",
                topic, str(message.data), new_base)

    def _on_stair_state(self, topic, message):
        with self._lock:
            self._stair_active_by_topic[topic] = (
                stair_transition_is_active(message.data))

    def _stair_active(self):
        with self._lock:
            return any(self._stair_active_by_topic.values())

    def _truth_pose(self, now=None):
        """Return a fresh truth position and yaw, else (None, None)."""
        if now is None:
            now = rospy.Time.now()
        with self._lock:
            truth = self._truth
            received_at = self._truth_received_at
        if truth is None or received_at is None:
            return None, None
        if (now - received_at).to_sec() > self._truth_maximum_age:
            return None, None
        return (
            (float(truth.position.x), float(truth.position.y),
             float(truth.position.z)),
            yaw_from_quaternion(truth.orientation))

    def _recovery_anchor(self, now=None, prefer_healthy=False):
        """Return a reset pose, preferring the last supported pose on a fall."""
        if prefer_healthy:
            with self._lock:
                healthy_anchor = self._healthy_truth_anchor
            if healthy_anchor is not None:
                return healthy_anchor[0], healthy_anchor[1], "healthy_gazebo_truth"
        position, yaw = self._truth_pose(now)
        if position is not None:
            return position, yaw, "gazebo_truth"
        with self._lock:
            pose = self._pose
        if pose is None:
            return None, None, None
        body_pose = pose.pose.pose
        return (
            (float(body_pose.position.x), float(body_pose.position.y),
             float(body_pose.position.z)),
            yaw_from_quaternion(body_pose.orientation),
            "fastlio_fallback")

    def _publish_joy(self, button_index):
        message = Joy()
        message.header.stamp = rospy.Time.now()
        message.axes = [0.0] * 8
        message.buttons = [0] * 12
        if 0 <= button_index < len(message.buttons):
            message.buttons[button_index] = 1
        self._joy_pub.publish(message)

    def _set_mapping_pause(self, active):
        active = bool(active)
        if active == self._mapping_pause_asserted:
            return
        self._mapping_pause_asserted = active
        # Stop commanded motion before the first suspect cloud can be
        # followed by another push; mapping pause follows in the same state
        # transition. Release reverses the order after healthy confirmation.
        if active:
            self._hold_pub.publish(Bool(data=True))
            self._recovery_state_pub.publish(Bool(data=True))
        else:
            self._recovery_state_pub.publish(Bool(data=False))
            self._hold_pub.publish(Bool(data=False))

    def _is_fallen(self):
        with self._lock:
            imu = self._imu
            pose = self._pose
            truth = self._truth
            twist = self._truth_twist
            floor_base = self._floor_base
            stair_active = any(self._stair_active_by_topic.values())
        if stair_active or imu is None:
            return False
        roll, pitch = roll_pitch_from_quaternion(imu.orientation)
        if truth is not None:
            z = float(truth.position.z)
        elif pose is not None:
            z = float(pose.pose.pose.position.z)
        else:
            return (abs(roll) > self._roll_limit or
                    abs(pitch) > self._pitch_limit)
        return relative_fall_evidence(
            z - floor_base, roll, pitch,
            self._roll_limit, self._pitch_limit, self._z_limit,
            self._flat_low_margin)

    def _record_healthy_truth_height(self):
        """Record health for release only; never infer a floor from it."""
        with self._lock:
            truth = self._truth
            twist = self._truth_twist
            floor_base = self._floor_base
            recovering = self._recovering
            stair_active = any(self._stair_active_by_topic.values())
        if truth is None or recovering or stair_active:
            return False
        roll, pitch = roll_pitch_from_quaternion(truth.orientation)
        relative_z = float(truth.position.z) - floor_base
        if (abs(roll) > self._recovery_release_tilt or
                abs(pitch) > self._recovery_release_tilt or
                relative_z < self._minimum_healthy_relative_z):
            # Belly-down can have near-zero tilt; the relative-height gate
            # keeps that posture from replacing the healthy reference.
            return False
        with self._lock:
            if (not self._recovering and
                    abs(self._floor_base - floor_base) < 1.0e-9):
                self._healthy_truth_z = float(truth.position.z)
                # A body can be upright for several frames while crossing a
                # stair-edge void. Only a settled pose is known to have support
                # and is therefore safe to reuse after a fall.
                if twist is not None and (
                        math.hypot(float(twist.linear.x),
                                   float(twist.linear.y)) <=
                        self._healthy_anchor_speed_limit and
                        abs(float(twist.angular.z)) <=
                        self._healthy_anchor_yaw_rate_limit):
                    self._healthy_truth_anchor = (
                        (float(truth.position.x), float(truth.position.y),
                         float(truth.position.z)),
                        yaw_from_quaternion(truth.orientation))
                return True
        return False

    def _recovery_target_z(self):
        with self._lock:
            base = (self._recovery_floor_base
                    if self._recovery_floor_base is not None
                    else self._floor_base)
        return float(base) + self._upright_z

    def _upright_in_place(self, position=None, yaw=None):
        source = "explicit_recovery_target"
        if position is None or yaw is None:
            position, yaw, source = self._recovery_anchor()
        if position is None or yaw is None:
            return False
        state = ModelState()
        state.model_name = "a1_gazebo"
        state.reference_frame = "world"
        state.pose.position.x = position[0]
        state.pose.position.y = position[1]
        state.pose.position.z = self._recovery_target_z()
        state.pose.orientation = quaternion_from_yaw(yaw)
        try:
            response = self._set_model_state(state)
            if not response.success:
                rospy.logwarn("set_model_state rejected (%s).", source)
            return bool(response.success)
        except rospy.ServiceException as error:
            rospy.logwarn("set_model_state failed: %s", error)
            return False

    def _arm_controller_local_reset(self, position, yaw):
        """Make one joy-10 reset consume the same floor-local Gazebo pose."""
        if position is None or yaw is None:
            return False
        rospy.set_param("/simenv/local_reset/x", float(position[0]))
        rospy.set_param("/simenv/local_reset/y", float(position[1]))
        rospy.set_param("/simenv/local_reset/z", self._recovery_target_z())
        rospy.set_param("/simenv/local_reset/yaw", float(yaw))
        rospy.set_param("/simenv/local_reset/use_startup_stance", False)
        # The controller consumes and clears this latch.  It must be armed
        # again for a bounded retry; otherwise the same joy-10 falls through
        # to its legacy startup pose and creates an unobserved map-frame jump.
        rospy.set_param("/simenv/local_reset/enabled", True)
        return True

    def _begin_recovery(self):
        now = rospy.Time.now()
        if (now - self._last_recover).to_sec() < self._cooldown:
            return False
        with self._lock:
            floor_base = self._floor_base
            healthy_truth_z = self._healthy_truth_z
            self._recovering = True
            self._recovery_floor_base = floor_base
            healthy_minimum = (
                floor_base + self._minimum_healthy_relative_z)
            healthy_maximum = (
                floor_base + self._upright_z +
                self._recovery_height_tolerance)
            if (healthy_truth_z is not None and
                    healthy_minimum <= healthy_truth_z <= healthy_maximum):
                self._recovery_expected_healthy_z = healthy_truth_z
            else:
                self._recovery_expected_healthy_z = (
                    floor_base + self._standing_body_height)
            self._recovery_last_truth_sequence = self._truth_sample_sequence
        self._recovery_retry_count = 0
        self._recovery_stuck = False
        self._recovery_count += 1
        self._set_mapping_pause(True)
        self._status_pub.publish(String(data="RECOVERY_ACTIVE"))
        self._recover_step = 0
        self._recover_step_time = now
        self._recovery_started_at = now
        self._recovery_release_streak = 0
        self._last_recover = now
        rospy.logwarn(
            "Fall confirmed; stand/RL recovery base=%.2f reset_z=%.2f "
            "gazebo_reset=%s.",
            floor_base, self._recovery_target_z(), self._use_gazebo_reset)

        if self._use_gazebo_reset:
            # Preserve the current physical XY exactly. Moving back to an
            # older healthy-position anchor while FAST-LIO is quarantined
            # creates an unobserved map-frame jump (fix14: 0.61 m), after
            # which an in-room pose can be reported as a corridor exit. The
            # last healthy yaw remains useful because roll/pitch collapse can
            # corrupt the fallen bodys projected yaw. Both Gazebo upright and
            # joy-10 controller reset must consume this identical target.
            position, current_yaw, _source = self._recovery_anchor(
                now, prefer_healthy=False)
            with self._lock:
                healthy_anchor = getattr(
                    self, "_healthy_truth_anchor", None)
            yaw = (float(healthy_anchor[1])
                   if healthy_anchor is not None else current_yaw)
            if position is not None and yaw is not None:
                self._arm_controller_local_reset(position, yaw)
                self._upright_in_place(position=position, yaw=yaw)
            else:
                self._upright_in_place()
            self._publish_joy(10)
        else:
            self._publish_joy(1)
        return True

    def _start_candidate_pause(self):
        with self._lock:
            floor_base = self._floor_base
            healthy_z = self._healthy_truth_z
            minimum_z = floor_base + self._minimum_healthy_relative_z
            maximum_z = (floor_base + self._upright_z +
                         self._recovery_height_tolerance)
            self._candidate_floor_base = floor_base
            self._candidate_expected_healthy_z = (
                healthy_z if healthy_z is not None and
                minimum_z <= healthy_z <= maximum_z
                else floor_base + self._standing_body_height)
            self._candidate_last_truth_sequence = self._truth_sample_sequence
        self._candidate_clear_streak = 0
        self._set_mapping_pause(True)

    def _consume_candidate_health_sample(self, now):
        with self._lock:
            truth = self._truth
            twist = self._truth_twist
            received_at = self._truth_received_at
            sequence = self._truth_sample_sequence
            expected_z = self._candidate_expected_healthy_z
            floor_base = self._candidate_floor_base
        if (truth is None or twist is None or received_at is None or
                expected_z is None or floor_base is None or
                sequence == self._candidate_last_truth_sequence):
            return False
        self._candidate_last_truth_sequence = sequence
        if (now - received_at).to_sec() > self._truth_maximum_age:
            self._candidate_clear_streak = 0
            return False
        roll, pitch = roll_pitch_from_quaternion(truth.orientation)
        linear_speed = math.sqrt(
            float(twist.linear.x) ** 2 +
            float(twist.linear.y) ** 2 +
            float(twist.linear.z) ** 2)
        angular_speed = math.sqrt(
            float(twist.angular.x) ** 2 +
            float(twist.angular.y) ** 2 +
            float(twist.angular.z) ** 2)
        stable = recovery_truth_sample_is_stable(
            float(truth.position.z), expected_z, roll, pitch,
            linear_speed, angular_speed,
            self._recovery_height_tolerance,
            self._recovery_release_tilt,
            self._recovery_release_linear_speed,
            self._recovery_release_angular_speed,
            float(truth.position.z) - float(floor_base),
            self._minimum_healthy_relative_z)
        self._candidate_clear_streak = (
            self._candidate_clear_streak + 1 if stable else 0)
        if self._candidate_clear_streak < self._candidate_clear_confirm_count:
            return False
        with self._lock:
            self._healthy_truth_z = float(truth.position.z)
            self._candidate_floor_base = None
            self._candidate_expected_healthy_z = None
        self._candidate_clear_streak = 0
        self._set_mapping_pause(False)
        return True

    def _consume_recovery_truth_sample(self, now):
        with self._lock:
            truth = self._truth
            twist = self._truth_twist
            received_at = self._truth_received_at
            sequence = self._truth_sample_sequence
            expected_z = self._recovery_expected_healthy_z
            floor_base = self._recovery_floor_base
        if (truth is None or twist is None or received_at is None or
                expected_z is None or
                sequence == self._recovery_last_truth_sequence):
            return False
        self._recovery_last_truth_sequence = sequence
        if (now - received_at).to_sec() > self._truth_maximum_age:
            self._recovery_release_streak = 0
            return False
        roll, pitch = roll_pitch_from_quaternion(truth.orientation)
        linear_speed = math.sqrt(
            float(twist.linear.x) ** 2 +
            float(twist.linear.y) ** 2 +
            float(twist.linear.z) ** 2)
        angular_speed = math.sqrt(
            float(twist.angular.x) ** 2 +
            float(twist.angular.y) ** 2 +
            float(twist.angular.z) ** 2)
        stable = recovery_truth_sample_is_stable(
            float(truth.position.z), expected_z, roll, pitch,
            linear_speed, angular_speed,
            self._recovery_height_tolerance,
            self._recovery_release_tilt,
            self._recovery_release_linear_speed,
            self._recovery_release_angular_speed,
            float(truth.position.z) - float(floor_base),
            self._minimum_healthy_relative_z)
        if stable:
            self._recovery_release_streak += 1
        else:
            self._recovery_release_streak = 0
        return (
            self._recovery_release_streak >=
            self._recovery_release_confirm_count)

    def _finish_recovery(self):
        # Only a fresh, consecutive truth-health gate can reach this method.
        with self._lock:
            if self._truth is not None:
                self._healthy_truth_z = float(self._truth.position.z)
            self._recovering = False
            self._recovery_stuck = False
            self._recovery_retry_count = 0
            self._recovery_floor_base = None
            self._recovery_expected_healthy_z = None
            self._candidate_floor_base = None
            self._candidate_expected_healthy_z = None
        self._recover_step = 0
        self._fall_streak = 0
        self._candidate_clear_streak = 0
        self._recovery_state_pub.publish(Bool(data=False))
        self._mapping_pause_asserted = False
        self._hold_pub.publish(Bool(data=False))
        self._status_pub.publish(String(data="RECOVERY_COMPLETE"))
        rospy.loginfo("Fall recovery truth-stable; resuming exploration.")

    def _restart_recovery_attempt(self, now):
        self._recovery_retry_count += 1
        self._recovery_stuck = False
        self._recover_step = 0
        self._recover_step_time = now
        self._recovery_started_at = now
        self._recovery_release_streak = 0
        with self._lock:
            self._recovery_last_truth_sequence = self._truth_sample_sequence
        self._set_mapping_pause(True)
        self._status_pub.publish(String(data="RECOVERY_RETRY_{}".format(
            self._recovery_retry_count)))
        if self._use_gazebo_reset:
            position, current_yaw, _source = self._recovery_anchor(
                now, prefer_healthy=False)
            with self._lock:
                healthy_anchor = self._healthy_truth_anchor
            yaw = (float(healthy_anchor[1])
                   if healthy_anchor is not None else current_yaw)
            if position is not None and yaw is not None:
                self._arm_controller_local_reset(position, yaw)
                self._upright_in_place(position=position, yaw=yaw)
            else:
                self._upright_in_place()
            self._publish_joy(10)
        else:
            self._publish_joy(1)
        rospy.logwarn(
            "Fall recovery attempt timed out; restarting bounded attempt %d/%d.",
            self._recovery_retry_count, self._recovery_retry_limit)


    def _advance_recovery(self, now):
        total_elapsed = (now - self._recovery_started_at).to_sec()
        if (not self._recovery_stuck and
                total_elapsed >= self._recovery_maximum_duration and
                self._recovery_retry_count < self._recovery_retry_limit):
            self._restart_recovery_attempt(now)
            return
        if (self._recovery_stuck or
                total_elapsed >= self._recovery_maximum_duration):
            if not self._recovery_stuck:
                self._recovery_stuck = True
                self._status_pub.publish(
                    String(data="RECOVERY_FAILED_STUCK"))
            # Fail closed: no truth-stable evidence means neither mapping nor
            # commanded motion may resume.
            self._set_mapping_pause(True)
            rospy.logerr_throttle(
                5.0, "RECOVERY_FAILED_STUCK: retaining map/cmd holds.")
            return
        elapsed = (now - self._recover_step_time).to_sec()
        if self._recover_step == 0 and elapsed >= 1.0:
            self._publish_joy(1)
            self._recover_step = 1
            self._recover_step_time = now
        elif self._recover_step == 1 and elapsed >= 4.0:
            self._publish_joy(3)
            self._recover_step = 2
            self._recover_step_time = now
            with self._lock:
                self._recovery_last_truth_sequence = (
                    self._truth_sample_sequence)
            self._recovery_release_streak = 0
        elif self._recover_step == 2:
            stable = self._consume_recovery_truth_sample(now)
            if stable and elapsed >= self._recovery_minimum_rl_seconds:
                self._finish_recovery()

    def _on_timer(self, _event):
        now = rospy.Time.now()
        if (now - self._node_start).to_sec() < self._start_grace:
            return
        with self._lock:
            recovering = self._recovering
        if recovering:
            self._advance_recovery(now)
            return

        if self._stair_active():
            # The stair manager now owns fall safety and command motion.
            self._fall_streak = 0
            self._candidate_clear_streak = 0
            self._candidate_floor_base = None
            self._candidate_expected_healthy_z = None
            self._set_mapping_pause(False)
            return

        if self._is_fallen():
            self._candidate_clear_streak = 0
            self._fall_streak += 1
            # First strong sample pauses mapping; confirm_count only gates
            # physical reset, preventing confirmation-window cloud pollution.
            if not self._mapping_pause_asserted:
                self._start_candidate_pause()
            else:
                self._set_mapping_pause(True)
            if self._fall_streak >= self._confirm_count:
                if self._begin_recovery():
                    self._fall_streak = 0
            return

        self._fall_streak = 0
        if self._mapping_pause_asserted:
            self._consume_candidate_health_sample(now)
            return
        self._record_healthy_truth_height()


if __name__ == "__main__":
    rospy.init_node("compliant_fall_recovery")
    CompliantFallRecovery()
    rospy.spin()
