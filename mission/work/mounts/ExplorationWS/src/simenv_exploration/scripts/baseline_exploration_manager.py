#!/usr/bin/env python3
"""FAST-LIO + FUEL-lite baseline with optional layout-free room scheduling."""

import csv
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import replace

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import Joy, PointCloud2
from std_msgs.msg import Bool, Float32, Float64, Int32, String
from simenv_exploration.srv import CheckTwinCylinder, CheckTwinCylinderRequest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_CORRIDOR_SEMANTIC_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "ufoexplorer_interface")
if _CORRIDOR_SEMANTIC_DIR not in sys.path:
    sys.path.insert(0, _CORRIDOR_SEMANTIC_DIR)
from baseline_planning_core import (  # noqa: E402
    CorridorBranchScheduler, DepthBreadthConfig, DepthBreadthScheduler,
    OccupancyGrid2D,
    astar_safe_path, corridor_side_frontier_eligible,
    corridor_line_lateral_error,
    room_exit_endpoint_tolerance_allowed,
    corridor_local_door_context_allowed,
    corridor_live_start_override_allowed,
    corridor_axis_motion_consistency,
    corridor_portal_from_reachable_path,
    corridor_wall_midpoint, corridor_width_profile_is_uniform,
    drop_reached_waypoint_prefix, execution_result_matches,
    execution_waypoints, segmented_execution_waypoints,
    local_region_coverage, nearest_reachable_goal,
    orient_axis_away_from_origin,
    rebase_corridor_line_laterally,
    recent_reverse_trajectory_anchors,
    side_region_component_match,
)
from goal_executor_core import quaternion_yaw  # noqa: E402
from lightweight_room_core import (  # noqa: E402
    LightweightRoomConfig, LightweightRoomScheduler, _ray_distance,
    _clearance, _ray_distance, _segment_has_occupied, _state,
    aperture_width, doorway_is_lateral_to_corridor, exit_timeout_for_path,
)
from scan_lite_path_refiner import (  # noqa: E402
    FootprintCheck, PathRefiner, RefinerConfig, path_yaws, polyline_length,
)
from telemetry_core import integrate_trajectory  # noqa: E402
from corridor_semantic_core import (  # noqa: E402
    detect_parallel_wall_corridor,
)


BASELINE_MODE = "baseline_fastlio_exploration"
DISABLED_FEATURES = (
    "enable_structured_exploration", "enable_room_detection",
    "enable_room_state_machine", "enable_door_detection",
    "enable_door_attention", "enable_door_commit", "enable_door_alignment",
    "enable_cross_door", "enable_room_information_gain", "enable_room_recovery",
)


def structured_executor_cancel(execution_result, reason):
    """Return a goal-identity cancellation accepted by goal_executor."""
    if not isinstance(execution_result, dict):
        return None
    executor_goal = execution_result.get("goal")
    try:
        payload = {
            "goal_sequence": int(execution_result.get("goal_sequence", -1)),
            "goal_stamp": float(execution_result["goal_stamp"]),
            "goal_x": float(executor_goal["x"]),
            "goal_y": float(executor_goal["y"]),
            "reason": str(reason),
        }
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(payload[key])
               for key in ("goal_stamp", "goal_x", "goal_y")):
        return None
    return payload


class LocalizationRecoveryWindow:
    """Bounded recovery episode driven by post-request health evidence.

    The class is deliberately ROS-free so the timing and confirmation
    contract can be unit-tested without constructing the exploration manager.
    A recovery has one fixed wall-clock deadline; retries never extend it.
    """

    WAITING = "waiting"
    REQUEST = "request"
    RECOVERED = "recovered"
    EXHAUSTED = "exhausted"

    def __init__(self, grace_sec, retry_interval_sec, maximum_attempts,
                 required_healthy_confirmations):
        self.grace_sec = max(0.1, float(grace_sec))
        self.retry_interval_sec = max(0.1, float(retry_interval_sec))
        self.maximum_attempts = max(1, int(maximum_attempts))
        self.required_healthy_confirmations = max(
            1, int(required_healthy_confirmations))
        self.reset()

    def reset(self):
        self.active = False
        self.started_at = None
        self.deadline = None
        self.last_request_at = None
        self.attempts = 0
        self.status_sequence_at_start = -1
        self.last_observed_status_sequence = -1
        self.grid_update_at_start = -1
        self.healthy_streak = 0

    def begin(self, now, status_sequence, grid_update_count):
        now = float(now)
        self.active = True
        self.started_at = now
        self.deadline = now + self.grace_sec
        self.last_request_at = None
        self.attempts = 0
        self.status_sequence_at_start = int(status_sequence)
        self.last_observed_status_sequence = int(status_sequence)
        self.grid_update_at_start = int(grid_update_count)
        self.healthy_streak = 0

    def observe_status(self, sequence, healthy, invalid_count):
        sequence = int(sequence)
        if (not self.active or
                sequence <= self.last_observed_status_sequence):
            return
        self.last_observed_status_sequence = sequence
        if (sequence > self.status_sequence_at_start and bool(healthy) and
                int(invalid_count) == 0):
            self.healthy_streak += 1
        else:
            self.healthy_streak = 0

    def poll(self, now, grid_update_count):
        if not self.active:
            return self.WAITING
        now = float(now)
        fresh_status = (
            self.last_observed_status_sequence >
            self.status_sequence_at_start)
        fresh_grid = int(grid_update_count) > self.grid_update_at_start
        if (fresh_status and fresh_grid and
                self.healthy_streak >=
                self.required_healthy_confirmations):
            return self.RECOVERED
        if now >= float(self.deadline):
            return self.EXHAUSTED
        retry_due = (
            self.last_request_at is None or
            now - self.last_request_at >= self.retry_interval_sec)
        if self.attempts < self.maximum_attempts and retry_due:
            self.attempts += 1
            self.last_request_at = now
            return self.REQUEST
        return self.WAITING


class BaselineExplorationManager:
    def __init__(self):
        self.output_dir = os.path.abspath(rospy.get_param("~output_dir"))
        self.map_file = os.path.abspath(rospy.get_param("~map_file"))
        self.statistics_file = os.path.abspath(rospy.get_param("~statistics_file"))
        self.task_mode = rospy.get_param("/task_mode", "")
        if self.task_mode != BASELINE_MODE:
            raise RuntimeError("task_mode must be " + BASELINE_MODE)
        enabled = [name for name in DISABLED_FEATURES
                   if bool(rospy.get_param("/" + name, False))]
        if enabled:
            raise RuntimeError("baseline forbids enabled features: " + ", ".join(enabled))

        self.maximum_duration = float(rospy.get_param("~maximum_duration", 400.0))
        self.maximum_goals = int(rospy.get_param("~maximum_goals", 40))
        self.startup_timeout = float(rospy.get_param("~startup_timeout", 120.0))
        self.planner_timeout = float(rospy.get_param("~planner_timeout", 35.0))
        self.goal_timeout = float(rospy.get_param("~goal_timeout", 180.0))
        self.clearance = float(rospy.get_param("~astar_clearance", 0.30))
        self.reached_tolerance = float(rospy.get_param("~reached_tolerance", 0.15))
        self.waypoint_spacing = float(rospy.get_param("~waypoint_spacing", 0.30))
        self.enable_scan_lite = bool(rospy.get_param(
            "~enable_scan_lite_refiner",
            rospy.get_param("/enable_scan_lite_refiner", False)))
        self.scan_failure_policy = rospy.get_param(
            "~scan_lite_failure_policy", "request_astar_replan")
        if self.scan_failure_policy not in (
                "fallback_original_path", "request_astar_replan", "fail_goal"):
            raise ValueError("invalid scan_lite_failure_policy")
        self.scan_config = RefinerConfig(
            body_front_offset=float(rospy.get_param("~body_front_offset", 0.06675)),
            body_rear_offset=float(rospy.get_param("~body_rear_offset", -0.06675)),
            body_collision_radius=float(rospy.get_param("~body_collision_radius", 0.11772)),
            body_min_height=float(rospy.get_param("~body_min_height", -0.057)),
            body_max_height=float(rospy.get_param("~body_max_height", 0.057)),
            body_safety_margin=float(rospy.get_param("~body_safety_margin", 0.05)),
            unknown_policy=rospy.get_param("~scan_lite_unknown_policy", "penalize"),
            refinement_search_radius=float(rospy.get_param("~refinement_search_radius", 0.60)),
            refinement_search_step=float(rospy.get_param("~refinement_search_step", 0.15)),
            enable_waypoint_simplification=bool(rospy.get_param(
                "~enable_waypoint_simplification", True)),
            waypoint_collinearity_tolerance=float(rospy.get_param(
                "~waypoint_collinearity_tolerance", 0.08)),
            waypoint_max_direct_connect_distance=float(rospy.get_param(
                "~waypoint_max_direct_connect_distance", 2.50)),
            waypoint_max_yaw_change=float(rospy.get_param(
                "~waypoint_max_yaw_change", 0.45)))
        self.scan_config.validate()
        self.minimum_map_updates = int(rospy.get_param("~minimum_map_updates", 1))
        # ENTRY already crossed a portal that passed A* and SCAN-lite.  Its
        # first in-room view should not wait on the slower statistics writer;
        # it is still replanned and preflight-checked on the latest grid.
        self.room_entry_replan_wait = float(rospy.get_param(
            "~room_entry_replan_wait_seconds", 1.0))
        # All paths are planned from ``self.grid`` rather than the slower
        # JSON map-statistics writer.  Limit the post-goal wait to one fresh
        # occupancy callback so planning does not idle for several seconds
        # merely waiting for telemetry to be flushed to disk.
        self.replan_grid_wait = float(rospy.get_param(
            "~replan_grid_wait_seconds", 0.75))
        self.minimum_free_voxels = int(rospy.get_param("~minimum_free_voxels", 5000))
        self.maximum_consecutive_failures = int(rospy.get_param(
            "~maximum_consecutive_failures", 3))
        self.trajectory_history_spacing = float(rospy.get_param(
            "~trajectory_history_spacing", 0.75))
        self.fuel_duplicate_radius = float(rospy.get_param(
            "~fuel_execution_duplicate_radius", 0.90))
        self.fuel_allow_transit_revisit_fallback = bool(rospy.get_param(
            "~fuel_allow_transit_revisit_fallback", True))
        self.fuel_minimum_selected_score = float(rospy.get_param(
            "~fuel_minimum_selected_score", -math.inf))
        self.preserve_planner_diagnostics = bool(rospy.get_param(
            "~preserve_planner_diagnostics", True))
        self.corridor_minimum_advance = float(rospy.get_param(
            "~corridor_minimum_advance", 2.5))
        self.corridor_recent_exclusion = float(rospy.get_param(
            "~corridor_recent_exclusion", 1.5))
        self.corridor_forward_distance = float(rospy.get_param(
            "~corridor_forward_distance", 5.0))
        # After the near doorway pair has been completed on F1, break the
        # next long corridor chord into shorter, still centreline-constrained
        # observations. A seven-metre chord can carry the camera completely
        # past the far doorway pair before a safe entry candidate is rebuilt.
        # This is disabled by default; the F1 launch enables it privately.
        self.corridor_far_pair_search_advance = float(rospy.get_param(
            "~corridor_far_pair_search_maximum_advance_m", 0.0))
        self.corridor_far_pair_search_after_exits = int(rospy.get_param(
            "~corridor_far_pair_search_after_exits", 2))
        # Before corridor geometry has accumulated enough updates to latch,
        # do not hand the robot back to unconstrained FUEL exploration in the
        # lobby. Use a short, reachable forward probe to acquire the corridor;
        # FUEL remains a fallback only when that ray is genuinely blocked.
        self.enable_corridor_acquisition = bool(rospy.get_param(
            "~enable_corridor_acquisition", True))
        self.corridor_acquisition_distance = float(rospy.get_param(
            "~corridor_acquisition_distance", 4.0))
        self.corridor_acquisition_minimum_advance = float(rospy.get_param(
            "~corridor_acquisition_minimum_advance", 1.50))
        self.corridor_acquisition_maximum_detour_ratio = float(rospy.get_param(
            "~corridor_acquisition_maximum_detour_ratio", 1.18))
        # A stair exit can face away from the upper-floor corridor.  In that
        # mode select the longest online known-free radial ray instead of
        # assuming the final stair-policy yaw is the corridor direction.
        self.corridor_acquisition_heading_search = bool(rospy.get_param(
            "~corridor_acquisition_heading_search", False))
        self.corridor_acquisition_heading_samples = int(rospy.get_param(
            "~corridor_acquisition_heading_samples", 16))
        self.corridor_acquisition_heading_range = float(rospy.get_param(
            "~corridor_acquisition_heading_range_m", 12.0))
        # Upper-floor stair exits normally open into a broad lobby rather than
        # directly into the corridor.  Rank acquisition motion with an online
        # parallel-wall semantic (long balanced walls + known-free centre),
        # falling back to radial free-space search until those walls enter the
        # local map.  No layout coordinates or trained model are used.
        self.corridor_acquisition_parallel_wall_semantic = bool(
            rospy.get_param(
                "~corridor_acquisition_parallel_wall_semantic", False))
        # Preferred acquisition heading (radians) supplied by the floor
        # handoff guide, which faces the robot into the corridor before
        # exploration starts.  In run R8k the F3 robot turned 180 deg during
        # the pre-sweep rescan and acquisition then selected the *arrival*
        # direction (south, already mapped) instead of the corridor forward
        # (north, unmapped), locking the corridor axis south and failing the
        # floor.  Ranking this guide heading above the ordinary online axis
        # keeps acquisition pointed down the real corridor until the sweep
        # establishes its own axis.
        preferred_heading = rospy.get_param(
            "~corridor_acquisition_preferred_heading", None)
        if preferred_heading is not None:
            try:
                preferred_heading = float(preferred_heading)
            except (TypeError, ValueError):
                preferred_heading = None
        self.corridor_acquisition_preferred_heading = preferred_heading
        # The parallel-wall relock detector rasterizes wall angle in 10-deg
        # steps, so a pure-N/S corridor can be reported as +-10 deg off-axis
        # (run R11: axis [-0.174,0.985] at y=10.4 pushed every corridor-sweep
        # candidate beyond the x=-1.1 wall, the only A* route went through
        # the room doorway, the robot wedged in the door and the sweep dead-
        # locked for 250 s).  When a preferred guide heading exists and the
        # relock axis is within this tolerance of it, snap the axis back to
        # the guide heading (the floors are axis-aligned by construction).
        self.corridor_axis_preferred_snap_tolerance_deg = float(
            rospy.get_param(
                "~corridor_axis_preferred_snap_tolerance_deg", 25.0))
        self.corridor_acquisition_semantic_search_radius = float(
            rospy.get_param(
                "~corridor_acquisition_semantic_search_radius_m", 12.0))
        self.corridor_acquisition_semantic_minimum_length = float(
            rospy.get_param(
                "~corridor_acquisition_semantic_minimum_length_m", 4.0))
        # Optional post-acquisition relock for a fresh upper-floor manager.
        # The stair/lobby-to-corridor turn can dominate recent-trajectory PCA
        # even after the robot is physically between the corridor walls.
        # Floor 1 keeps the historical path (default false); floor 2 may use
        # the already-online parallel-wall detector once, at entry lock.
        self.corridor_entry_axis_relock_parallel_wall_semantic = bool(
            rospy.get_param(
                "~corridor_entry_axis_relock_parallel_wall_semantic", False))
        # A sparse upper-floor rolling map can mistake a doorway/furniture
        # edge for the long corridor walls.  When enabled, accept a semantic
        # relock only when it agrees with the online displacement already
        # travelled from this floor's mission origin.  Defaults are inert so
        # the proven floor-1 corridor policy is byte-for-byte behavioural.
        self.corridor_entry_axis_relock_maximum_motion_angle = math.radians(
            float(rospy.get_param(
                "~corridor_entry_axis_relock_maximum_motion_angle_deg",
                180.0)))
        self.corridor_entry_axis_relock_minimum_motion = float(
            rospy.get_param(
                "~corridor_entry_axis_relock_minimum_motion_m", 2.0))
        # Once the robot has actually crossed from the lobby into an online
        # corridor-shaped region, keep exploration one-way even if a later
        # rolling map window briefly loses the corridor classifier.  This is
        # deliberately a short relative displacement, not a world/layout
        # coordinate, and is released only by the existing terminal-return
        # logic.
        self.corridor_entry_forward_lock_distance = float(rospy.get_param(
            "~corridor_entry_forward_lock_distance_m", 1.0))
        self.corridor_reversal_limit = int(rospy.get_param(
            "~corridor_reversal_limit", 1))
        # Long centreline goals can be cut by one or two isolated projection
        # cells even though the currently observed corridor is traversable.
        # In that case advance by a short A*-verified increment and search a
        # narrow lateral band, allowing the rolling map to grow forward.
        self.corridor_forward_recovery_minimum_advance = float(
            rospy.get_param(
                "~corridor_forward_recovery_minimum_advance_m", 0.8))
        self.corridor_forward_lateral_search = float(rospy.get_param(
            "~corridor_forward_lateral_search_m", 0.40))
        self.corridor_no_frontier_probe_after = int(rospy.get_param(
            "~corridor_no_frontier_probe_after", 2))
        self.corridor_no_frontier_probe_distance = float(rospy.get_param(
            "~corridor_no_frontier_probe_distance_m", 0.45))
        # Optional upper-floor escape from a stale 2-D projection. Normal
        # corridor motion still requires A*. When enabled by the isolated F2
        # process, repeated A* exhaustion may advance one short centreline
        # segment only after densely sampled independent 3-D body footprints
        # are all collision-free and no transverse terminal wall is visible.
        # The default is false, so F1 behaviour is unchanged.
        self.enable_corridor_monotonic_forward_probe = bool(rospy.get_param(
            "~enable_corridor_monotonic_forward_probe", False))
        self.corridor_monotonic_probe_after = int(rospy.get_param(
            "~corridor_monotonic_probe_after", 2))
        self.corridor_monotonic_probe_distance = float(rospy.get_param(
            "~corridor_monotonic_probe_distance_m", 0.60))
        self.corridor_monotonic_probe_sample_spacing = float(rospy.get_param(
            "~corridor_monotonic_probe_sample_spacing_m", 0.10))
        self.corridor_monotonic_probe_max_unknown_queries = int(
            rospy.get_param(
                "~corridor_monotonic_probe_max_unknown_queries", 8))
        self.corridor_monotonic_probe_limit = int(rospy.get_param(
            "~corridor_monotonic_probe_limit", 24))
        # If the forward corridor has genuinely ended but a side room was
        # missed, make one constrained reverse pass to observe its doorway
        # from the opposite direction.  This is deliberately not a return-to-
        # lobby/F1 command: ordinary room completion is still required for
        # that handoff.
        self.enable_terminal_missing_room_retrace = bool(rospy.get_param(
            "~enable_terminal_missing_room_retrace", True))
        self.terminal_missing_room_rescan_limit = int(rospy.get_param(
            "~terminal_missing_room_rescan_limit", 2))
        # A shortened local map can contain occupied returns several metres
        # ahead while the corridor still continues around them.  Only a
        # genuinely close transverse wall may start the missing-room reverse
        # pass.  If the pass was caused by repeated frontier loss instead,
        # bound how far it may travel so it can never become a return to F1.
        self.terminal_missing_room_retrace_wall_distance = float(
            rospy.get_param(
                "~terminal_missing_room_retrace_wall_distance_m", 2.25))
        self.terminal_missing_room_retrace_max_distance = float(
            rospy.get_param(
                "~terminal_missing_room_retrace_max_distance_m", 8.0))
        # A floor-specific recovery may enable the bounded reverse pass only
        # after enough outbound corridor has actually been traversed.  The
        # default preserves the original F1 behaviour; F2 uses this guard to
        # avoid treating a room-row corner near the first doors as a terminal.
        self.terminal_missing_room_retrace_minimum_outbound_progress = float(
            rospy.get_param(
                "~terminal_missing_room_retrace_minimum_outbound_progress_m",
                0.0))
        # Failed doorway landmarks are retained for duplicate suppression.
        # An upper-floor mission must reserve enough time to return to its
        # corridor entrance and hand control to the next staircase even when
        # one or more doors cannot be entered safely. The progress trigger is
        # online station distance; the time trigger is a final continuity
        # reserve. Both default to zero so F1 keeps its validated policy.
        self.corridor_partial_return_minimum_outbound_progress = float(
            rospy.get_param(
                "~corridor_partial_return_minimum_outbound_progress_m", 0.0))
        self.corridor_partial_return_reserve = float(rospy.get_param(
            "~corridor_partial_return_reserve_seconds", 0.0))
        self.corridor_partial_return_room_minimum_remaining = float(
            rospy.get_param(
                "~corridor_partial_return_room_minimum_remaining_seconds", 0.0))
        self.corridor_partial_return_trigger_reason = None

        # A floor-specific manager may safely retry a nearby, cooled landmark
        # before spending another long corridor leg.  The default is disabled
        # so the validated first-floor route is unchanged.
        self.enable_known_unvisited_door_retry = bool(rospy.get_param(
            "~enable_known_unvisited_door_retry", False))
        self.known_unvisited_door_retry_distance = float(rospy.get_param(
            "~known_unvisited_door_retry_distance_m", 6.5))
        self.known_unvisited_door_retry_only_on_terminal_return = bool(
            rospy.get_param(
                "~known_unvisited_door_retry_only_on_terminal_return", False))
        # Likewise, an upper floor can prevent a geometry-only terminal
        # return while a concrete unvisited doorway is still known.  This is
        # distinct from an entirely unseen room, for which the historical
        # bounded terminal-return policy remains available.
        self.terminal_return_blocked_by_known_unvisited_door = bool(
            rospy.get_param(
                "~terminal_return_blocked_by_known_unvisited_door", False))
        self.terminal_known_door_block_logged = False
        # A terminal wall shortens the rolling occupancy window, so the
        # generic corridor classifier can become false exactly beside the
        # final doorway pair.  Retain the already established online axis for
        # a small, bounded terminal region; local doors still need their
        # ordinary temporal, A*, SCAN-lite, width and lateral-normal checks.
        self.terminal_door_context_fallback_drawdown = float(rospy.get_param(
            "~terminal_door_context_fallback_drawdown_m", 1.75))
        self.terminal_door_context_fallback_lateral = float(rospy.get_param(
            "~terminal_door_context_fallback_lateral_m", 1.10))
        self.enable_terminal_door_context_fallback = bool(rospy.get_param(
            "~enable_terminal_door_context_fallback", True))
        # Executor success only proves that the map-frame EXIT endpoint was
        # reached.  A drifted or oblique door plane can leave that endpoint
        # physically inside the room.  A floor-specific instance may
        # therefore require the established station band and/or live,
        # bilateral corridor geometry before ROOM_EXITED is committed.  The
        # defaults remain false; launch files opt in only for the floor whose
        # route is being protected.
        self.room_exit_require_raw_corridor_confirmation = bool(
            rospy.get_param(
                "~room_exit_require_raw_corridor_confirmation", False))
        self.room_exit_require_station_corridor_confirmation = bool(
            rospy.get_param(
                "~room_exit_require_station_corridor_confirmation", False))
        self.corridor_confirmation_updates = int(rospy.get_param(
            "~corridor_confirmation_updates", 3))
        self.corridor_local_door_entry_lock_minimum_progress = float(
            rospy.get_param(
                "~corridor_local_door_entry_lock_minimum_progress_m", 2.0))
        self.corridor_minimum_visible_length = float(rospy.get_param(
            "~corridor_minimum_visible_length", 4.0))
        self.corridor_minimum_aspect_ratio = float(rospy.get_param(
            "~corridor_minimum_aspect_ratio", 1.60))
        self.corridor_uniform_width_sample_offset = float(rospy.get_param(
            "~corridor_uniform_width_sample_offset", 1.50))
        self.corridor_maximum_width_spread = float(rospy.get_param(
            "~corridor_maximum_width_spread", 0.90))
        self.corridor_side_coverage_priority = bool(rospy.get_param(
            "~enable_corridor_side_coverage_priority", True))
        self.corridor_side_entry_depth = float(rospy.get_param(
            "~corridor_side_entry_depth", 2.10))
        self.corridor_side_staging_minimum_depth = float(rospy.get_param(
            "~corridor_side_staging_minimum_depth", 1.05))
        self.corridor_side_station_search_radius = float(rospy.get_param(
            "~corridor_side_station_search_radius", 1.20))
        self.corridor_side_station_search_step = float(rospy.get_param(
            "~corridor_side_station_search_step", 0.30))
        self.corridor_side_raw_frontier_minimum_size = int(rospy.get_param(
            "~corridor_side_raw_frontier_minimum_size", 40))
        self.corridor_side_frontier_lateral = float(rospy.get_param(
            "~corridor_side_frontier_minimum_lateral", 1.20))
        self.corridor_side_maximum_longitudinal = float(rospy.get_param(
            "~corridor_side_frontier_maximum_longitudinal", 3.00))
        self.corridor_centerline_membership_tolerance = float(rospy.get_param(
            "~corridor_centerline_membership_tolerance", 0.75))
        # EXIT completion may stop within the executor's goal tolerance just
        # outside the tighter transit centreline band.  Keep a separate,
        # floor-configurable limit so normal corridor/door decisions remain
        # unchanged.
        self.room_exit_station_corridor_tolerance = float(rospy.get_param(
            "~room_exit_station_corridor_tolerance_m",
            self.corridor_centerline_membership_tolerance))
        # The initial station line may be latched from oblique lobby/door
        # geometry.  A physically completed room exit is much stronger
        # centreline evidence.  Permit a bounded lateral-only correction;
        # the axis and every remembered longitudinal door station stay fixed.
        self.corridor_exit_centerline_rebase_maximum = float(rospy.get_param(
            "~corridor_exit_centerline_rebase_maximum_m", 2.50))
        self.corridor_station_centerline_rebased = False
        self.corridor_station_centerline_calibrated = False
        self.enable_corridor_branch_scheduler = bool(rospy.get_param(
            "~enable_corridor_branch_scheduler", True))
        self.corridor_branch_station_tolerance = float(rospy.get_param(
            "~corridor_branch_station_tolerance", 1.75))
        self.corridor_branch_retry_cooldown = float(rospy.get_param(
            "~corridor_branch_retry_cooldown_seconds", 45.0))
        self.corridor_branch_entry_tolerance = float(rospy.get_param(
            "~corridor_branch_entry_target_tolerance", 0.90))
        self.enable_semantic_side_door_candidate = bool(rospy.get_param(
            "~enable_semantic_side_door_candidate", True))
        self.semantic_side_door_minimum_observations = int(rospy.get_param(
            "~semantic_side_door_minimum_observations", 3))
        self.semantic_side_door_minimum_penetration = float(rospy.get_param(
            "~semantic_side_door_minimum_penetration", 1.35))
        self.visited_region_component_minimum_lateral = float(rospy.get_param(
            "~visited_region_component_minimum_lateral", 0.75))
        self.visited_region_sample_spacing = float(rospy.get_param(
            "~visited_region_sample_spacing", 0.75))
        self.visited_region_maximum_station_separation = float(
            rospy.get_param(
                "~visited_region_maximum_station_separation", 6.0))
        self.region_return_relocation_radius = float(rospy.get_param(
            "~region_return_relocation_radius", 1.50))
        self.stair_return_relocation_radius = float(rospy.get_param(
            "~stair_return_relocation_radius", 2.40))
        self.stair_return_breadcrumb_spacing = float(rospy.get_param(
            "~stair_return_breadcrumb_spacing", 1.20))
        self.stair_return_centerline_tolerance = float(rospy.get_param(
            "~stair_return_centerline_tolerance", 0.85))
        # Room exploration keeps its configured budget; an already-authorized
        # stair return gets a separate bounded completion window.
        self.stair_return_grace_seconds = float(rospy.get_param(
            "~stair_return_grace_seconds", 90.0))
        self.region_exit_recovery_radius = float(rospy.get_param(
            "~region_exit_recovery_radius", 3.00))
        self.region_exit_recovery_limit = int(rospy.get_param(
            "~region_exit_recovery_limit", 2))
        self.region_stall_backtrack_distance = float(rospy.get_param(
            "~region_stall_backtrack_distance", 1.20))
        self.region_stall_backtrack_spacing = float(rospy.get_param(
            "~region_stall_backtrack_spacing", 0.30))
        self.region_stall_backtrack_limit = int(rospy.get_param(
            "~region_stall_backtrack_limit", 3))
        # A corridor endpoint can become occupied only after the robot arrives:
        # the rolling voxel projection incorporates a new wall/self-return and
        # SCAN-lite then rejects waypoint zero of every following path.  This
        # needs its own recent-trajectory escape; region backtracking is
        # intentionally disabled by the one-way corridor lock.
        self.corridor_start_collision_backtrack_distance = float(
            rospy.get_param(
                "~corridor_start_collision_backtrack_distance", 0.90))
        self.corridor_start_collision_backtrack_spacing = float(
            rospy.get_param(
                "~corridor_start_collision_backtrack_spacing", 0.25))
        self.corridor_start_collision_backtrack_limit = int(
            rospy.get_param(
                "~corridor_start_collision_backtrack_limit", 2))
        self.post_entry_lidar_maximum_rays = int(rospy.get_param(
            "~post_entry_lidar_maximum_rays", 1600))
        self.room_target_count = int(rospy.get_param("~room_target_count", 4))
        self.room_phase_target_seconds = float(rospy.get_param(
            "~room_phase_target_seconds", 160.0))
        self.enforce_room_phase_target_deadline = bool(rospy.get_param(
            "~enforce_room_phase_target_deadline", False))
        self.room_mission_entry_minimum_remaining = float(rospy.get_param(
            "~room_mission_entry_minimum_remaining_seconds", 24.0))
        # A SCAN-lite refinement can fail while the doorway map is still
        # changing, even though its LiDAR candidate is valid.  Permit one
        # quick retry after the map has refreshed; repeated failures fall
        # back to the normal long door cooldown.
        self.room_entry_refinement_retry_cooldown = float(rospy.get_param(
            "~room_entry_refinement_retry_cooldown_seconds", 12.0))
        self.room_entry_refinement_failures = {}
        self.stop_after_all_room_exits = bool(rospy.get_param(
            "~stop_after_all_room_exits", True))
        self.room_exit_terminal_wall_distance = float(rospy.get_param(
            "~room_exit_terminal_wall_distance_m", 8.0))
        # A return pass may start only from a *nearby* observed end wall.  The
        # latch is retained while travelling back, because the end wall will
        # naturally leave the forward five-metre LiDAR window on the first
        # return step.
        self.corridor_terminal_return_distance = float(rospy.get_param(
            "~corridor_terminal_return_distance_m", 5.0))
        self.corridor_terminal_return_latched = False
        # G2 reached while rooms are still missing: instead of failing
        # immediately, unlatch the terminal return once (retry limit) so the
        # corridor planner gets another exploration pass (fresh doors may
        # have come off cooldown or the reverse observation pass may reveal
        # an unseen portal).  Progress (exited-room count increase) resets
        # the counter; only repeated no-progress returns terminate.
        self.g2_missing_rooms_retries = 0
        self.g2_missing_rooms_retry_limit = int(rospy.get_param(
            "~g2_missing_rooms_retry_limit", 2))
        self._g2_last_exited_count = 0
        # Published once when either successful room completion or a
        # geometry-confirmed terminal-wall return starts.  The stair manager
        # uses this state to arm its independent Gazebo-truth lobby gate.
        # Without the latch, a partial (but valid) terminal return could pass
        # the physical stair entrance while F1 continued chasing a drifted
        # FAST-LIO waypoint.
        self.stair_return_transit_announced = False
        # Once the required rooms have been completed, spend a small, bounded
        # number of cycles surveying the currently detected corridor ahead.
        # This is map-frame geometry only: it does not use the layout or a
        # prescribed return route.  If the forward free corridor terminates,
        # the normal mission-home goal is selected.
        self.enable_post_room_corridor_sweep = bool(rospy.get_param(
            "~enable_post_room_corridor_sweep", True))
        self.post_room_corridor_sweep_max_goals = int(rospy.get_param(
            "~post_room_corridor_sweep_max_goals", 3))
        self.post_room_corridor_sweep_attempts = 0
        self.post_room_corridor_sweep_done = False
        self.room_new_door_maximum_lookbehind = float(rospy.get_param(
            "~room_new_door_maximum_lookbehind", 1.0))
        # A paired doorway may be revealed immediately after clearing an
        # exit, but it must be at the same physical corridor station.  This
        # tighter bound is deliberately independent from the generic
        # look-behind limit: the latter is only a rejection threshold, not
        # permission to reverse corridor exploration.
        self.paired_opposite_maximum_backtrack = float(rospy.get_param(
            "~paired_opposite_maximum_backtrack_m", 2.0))
        # Keep confirmed opposite-door evidence through the recenter transit.
        # F1 retains the historical 8 s default; fresh upper-floor managers
        # may widen it without changing downstairs scheduling.
        self.paired_opposite_recenter_pending_timeout = float(rospy.get_param(
            "~paired_opposite_recenter_pending_timeout_sec", 8.0))
        self.room_door_maximum_corridor_alignment = float(rospy.get_param(
            "~room_door_maximum_corridor_alignment", 0.55))
        # Optional online-geometry guard for upper-floor managers.  A false
        # opening formed by rolling-map free space can sit almost exactly on
        # the established corridor centreline; a physical side door cannot.
        # The default disables this gate so the already validated F1 policy
        # remains byte-for-byte equivalent at runtime.
        self.room_door_minimum_centerline_lateral = max(0.0, float(
            rospy.get_param(
                "~room_door_minimum_centerline_lateral_m", 0.0)))
        # A side doorway observed several metres ahead is necessarily viewed
        # obliquely.  Keep the map-derived corridor-side detector strict, but
        # allow the robot-local detector's independently A*/SCAN-validated
        # aperture to deviate farther from a perfect 90 degree normal.  Run25
        # repeatedly confirmed the far-room portal at |dot(n,axis)|=0.768;
        # the former shared 0.55 gate silently discarded every observation.
        self.local_door_maximum_corridor_alignment = float(rospy.get_param(
            "~local_door_maximum_corridor_alignment", 0.82))
        self.mission_home_tolerance = float(rospy.get_param(
            "~mission_home_tolerance", 0.50))
        self.stair_wait_tolerance = float(rospy.get_param(
            "~stair_wait_tolerance_m", 0.60))
        # With the temporary truth stair-entry guide enabled, F1 is a useful
        # return reference but not a necessary waypoint: once the return pass
        # has left the corridor, the stair controller can take the shorter
        # direct route to the verified entry.
        self.stair_handoff_on_corridor_exit = bool(rospy.get_param(
            "~stair_handoff_on_corridor_exit", False))
        self.stair_return_escape_done = False
        self.enable_local_doorway_detector = bool(rospy.get_param(
            "~enable_local_doorway_detector", True))
        self.local_candidate_freshness = float(rospy.get_param(
            "~local_candidate_freshness_seconds", 4.0))
        self.local_candidate_recent_window = float(rospy.get_param(
            "~local_candidate_recent_window_seconds", 20.0))
        self.local_door_minimum_unknown = float(rospy.get_param(
            "~local_door_minimum_unknown_behind_m2", 1.0))
        self.local_door_minimum_free = float(rospy.get_param(
            "~local_door_minimum_free_behind_m2", 6.0))
        self.local_door_maximum_semantic_width = float(rospy.get_param(
            "~local_door_maximum_semantic_width_m", 1.50))
        self.local_door_forward_high_water_tolerance = float(rospy.get_param(
            "~local_door_forward_high_water_tolerance_m", 1.0))
        self.corridor_new_branch_maximum_lookbehind = float(rospy.get_param(
            "~corridor_new_branch_maximum_lookbehind_m", 0.75))
        self.corridor_door_minimum_progress = float(rospy.get_param(
            "~corridor_door_detection_minimum_progress_m", 0.0))
        self.corridor_door_forward_commit_distance = float(rospy.get_param(
            "~corridor_door_forward_commit_distance_m", 1.50))
        # Normally only a corridor established from local-door evidence gets
        # the short pre-arm transit.  A fresh F2 manager, however, starts at a
        # GT-verified corridor entrance and can establish from raw parallel
        # walls before the first door becomes visible.  Allow that instance to
        # request the same short commit without changing the F1 default.
        self.corridor_short_door_commit_after_establishment = bool(
            rospy.get_param(
                "~corridor_short_door_commit_after_establishment", False))
        # A corridor-shaped scan can also occur in the lobby.  Require a
        # short, local forward confirmation after the corridor latch before a
        # side opening is allowed to preempt the forward goal.  This is
        # odometry-relative and does not use truth/layout coordinates.
        self.corridor_door_post_latch_motion = float(rospy.get_param(
            "~corridor_door_post_latch_motion_m", 2.0))
        self.first_room_minimum_corridor_station = float(rospy.get_param(
            "~first_room_minimum_corridor_station_m", 2.0))
        # Reject only locally inferred corridor-side doors implausibly far
        # ahead of the robot.  Infinity retains the existing generic/F1 path;
        # F2 supplies a finite local observation window to suppress the
        # repeated end-wall candidate seen in run55.
        self.corridor_side_candidate_max_forward_station = float(
            rospy.get_param(
                "~corridor_side_candidate_max_forward_station_m", math.inf))
        self.termination_unknown_area_threshold = float(rospy.get_param(
            "~termination_local_unknown_area_m2", 3.0))
        self.local_rescan_angle = float(rospy.get_param(
            "~local_rescan_angle_rad", 6.283185307))
        self.local_rescan_speed = float(rospy.get_param(
            "~local_rescan_angular_speed", 0.16))
        # A local rescan is an opportunistic observation action.  It must
        # never consume the remaining mission budget when the executor does
        # not return a completion message (for example while FAST-LIO is
        # briefly recovering at a doorway).
        self.local_rescan_maximum_seconds = float(rospy.get_param(
            "~local_rescan_maximum_seconds", 8.0))
        self.room_visual_sweep_maximum_seconds = float(rospy.get_param(
            "~room_visual_sweep_maximum_seconds", 8.0))
        self.local_rescan_cooldown = float(rospy.get_param(
            "~local_rescan_cooldown_seconds", 8.0))
        self.room_camera_sweep_enabled = bool(rospy.get_param(
            "~room_camera_sweep_enabled", True))
        self.room_camera_sweep_angle = float(rospy.get_param(
            "~room_camera_sweep_angle_rad", 0.5 * math.pi))
        self.room_camera_sweep_speed = float(rospy.get_param(
            "~room_camera_sweep_angular_speed", 0.60))
        self.room_camera_sweep_max_per_room = int(rospy.get_param(
            "~room_camera_sweep_max_per_room", 2))
        # The timed route already budgets a G1 LiDAR coverage stop in every
        # room.  Turn there once so RGB-D receives a deterministic inward
        # fan even when the asynchronous coverage mapper publishes late.
        # This is a rotation-only action, never a room detour.
        self.room_visual_force_g1_fan = bool(rospy.get_param(
            "~room_visual_force_g1_fan", True))
        self.room_visual_coverage_target = max(.4, min(1.0, float(
            rospy.get_param("~room_visual_coverage_target", .80))))
        self.room_visual_primary_min_angle = max(math.pi / 6.0, min(
            2.0 * math.pi, float(rospy.get_param(
                "~room_visual_primary_min_angle_rad", 1.55))))
        self.room_visual_occlusion_sweep_angle = max(math.pi / 6.0, min(
            math.pi, float(rospy.get_param(
                "~room_visual_occlusion_sweep_angle_rad", 1.55))))
        self.room_visual_completion_angle = max(math.pi / 6.0, min(
            math.pi, float(rospy.get_param(
                "~room_visual_completion_angle_rad", 1.10))))
        self.room_visual_deepening_enabled = bool(rospy.get_param(
            "~room_visual_deepening_enabled", True))
        self.room_visual_deepening_min_remaining = float(rospy.get_param(
            "~room_visual_deepening_min_remaining_s", 18.0))
        self.room_visual_deepening_budget = max(0, int(rospy.get_param(
            "~room_visual_deepening_budget", 0)))
        self.room_visual_deepening_min_completed_rooms = max(0, int(
            rospy.get_param("~room_visual_deepening_min_completed_rooms", 3)))
        self.room_visual_breadth_max_views = max(1, min(2, int(rospy.get_param(
            "~room_visual_breadth_max_views", 2))))
        self.room_visual_deepened_rooms = set()
        # A normal short fan is free of translational detours.  If it yielded
        # no geometry-validated red candidate at all, one opposite fan can
        # expose the previously unseen side without turning every room visit
        # into a full rotation.  Preserve a hard exit-time reserve.
        self.empty_visual_supplement_enabled = bool(rospy.get_param(
            "~empty_visual_supplement_enabled", True))
        self.empty_visual_supplement_angle = float(rospy.get_param(
            "~empty_visual_supplement_angle_rad", .90))
        self.empty_visual_supplement_minimum_remaining = float(rospy.get_param(
            "~empty_visual_supplement_minimum_remaining_s", 8.0))
        self.room_visual_completion_room_budget = max(0, int(rospy.get_param(
            "~room_visual_completion_room_budget", self.room_target_count)))
        self.room_visual_completion_minimum_rooms = max(0, int(rospy.get_param(
            "~room_visual_completion_minimum_rooms", 1)))
        self.room_visual_pending_strict_nudge_enabled = bool(rospy.get_param(
            "~room_visual_pending_strict_nudge_enabled", False))
        # Diagnostic high-recall profile: G1 and one LiDAR/A*-validated side
        # view each receive a complete in-place camera revolution.  It is
        # intentionally opt-in because its time cost is incompatible with the
        # normal 180-second throughput profile.
        self.room_visual_two_pose_full_sweep_enabled = bool(rospy.get_param(
            "~room_visual_two_pose_full_sweep_enabled", False))
        self.room_visual_full_sweep_angle = max(math.pi, min(
            2.0 * math.pi, float(rospy.get_param(
                "~room_visual_full_sweep_angle_rad", 2.0 * math.pi))))
        # With two laterally separated poses, the first pose supplies the
        # complete azimuth sweep.  The second pose is retained for parallax
        # and obstacle-occlusion recovery, but does not need to repeat every
        # already observed bearing.
        self.room_visual_secondary_sweep_angle = max(
            0.5 * math.pi, min(self.room_visual_full_sweep_angle, float(
                rospy.get_param("~room_visual_secondary_sweep_angle_rad",
                                self.room_visual_full_sweep_angle))))
        # In the timed profile the second camera centre is an occlusion
        # recovery action, not a compulsory room traverse.  The first
        # central full turn decides whether it is genuinely needed.
        self.room_visual_second_view_min_coverage = max(0.0, min(
            1.0, float(rospy.get_param(
                "~room_visual_second_view_min_coverage", 0.70))))
        self.room_visual_second_view_budget = max(0, int(rospy.get_param(
            "~room_visual_second_view_budget", 1)))
        self.room_visual_second_view_rooms = set()
        # Compact, isolated LiDAR returns inside an entered room are not
        # hazards by themselves.  They are only inexpensive cues for aiming
        # the RGB-D fan at a potential small object rather than sweeping an
        # arbitrary missing camera sector.
        self.lidar_visual_cue_enabled = bool(rospy.get_param(
            "~lidar_visual_cue_enabled", True))
        self.lidar_visual_cue_min_range = float(rospy.get_param(
            "~lidar_visual_cue_min_range_m", .70))
        self.lidar_visual_cue_max_range = float(rospy.get_param(
            "~lidar_visual_cue_max_range_m", 6.5))
        self.lidar_visual_cue_cell_size = float(rospy.get_param(
            "~lidar_visual_cue_cell_size_m", .18))
        self.lidar_visual_cue_max_cells = int(rospy.get_param(
            "~lidar_visual_cue_max_cells", 9))
        self.lidar_visual_cue_height_min = float(rospy.get_param(
            "~lidar_visual_cue_height_min_m", -.25))
        self.lidar_visual_cue_height_max = float(rospy.get_param(
            "~lidar_visual_cue_height_max_m", .85))
        self.room_visual_one_sector_completion_angle = float(rospy.get_param(
            "~room_visual_one_sector_completion_angle_rad", .70))
        self.room_visual_one_sector_completion_enabled = bool(rospy.get_param(
            "~room_visual_one_sector_completion_enabled", False))
        if (min(self.corridor_minimum_advance,
                self.corridor_recent_exclusion,
                self.corridor_forward_distance) <= 0.0 or
                self.corridor_forward_distance < self.corridor_minimum_advance or
                self.corridor_acquisition_distance <
                self.corridor_acquisition_minimum_advance or
                self.corridor_acquisition_minimum_advance <= 0.0 or
                self.corridor_acquisition_maximum_detour_ratio < 1.0 or
                min(self.corridor_acquisition_semantic_search_radius,
                    self.corridor_acquisition_semantic_minimum_length) <= 0.0 or
                self.corridor_entry_forward_lock_distance <= 0.0 or
                self.corridor_reversal_limit < 0 or
                self.corridor_forward_recovery_minimum_advance <= 0.0 or
                self.corridor_forward_recovery_minimum_advance >
                self.corridor_minimum_advance or
                self.corridor_forward_lateral_search < 0.0 or
                self.terminal_missing_room_rescan_limit < 1 or
                min(self.terminal_missing_room_retrace_wall_distance,
                    self.terminal_missing_room_retrace_max_distance) <= 0.0 or
                self.terminal_missing_room_retrace_minimum_outbound_progress < 0.0 or
                self.corridor_door_forward_commit_distance <= 0.0 or
                self.corridor_partial_return_minimum_outbound_progress < 0.0 or
                self.corridor_partial_return_reserve < 0.0 or
                self.corridor_partial_return_reserve > self.maximum_duration or
                self.corridor_partial_return_room_minimum_remaining < 0.0 or
                self.corridor_partial_return_room_minimum_remaining >
                self.maximum_duration or
                self.corridor_side_candidate_max_forward_station <= 0.0 or
                self.corridor_confirmation_updates < 1 or
                self.corridor_local_door_entry_lock_minimum_progress <= 0.0 or
                min(self.corridor_minimum_visible_length,
                    self.corridor_minimum_aspect_ratio,
                    self.corridor_uniform_width_sample_offset,
                    self.corridor_maximum_width_spread) <= 0.0 or
                min(self.corridor_side_entry_depth,
                    self.corridor_side_staging_minimum_depth,
                    self.corridor_side_station_search_step,
                    self.corridor_side_frontier_lateral,
                    self.corridor_side_maximum_longitudinal,
                    self.corridor_centerline_membership_tolerance,
                    self.room_exit_station_corridor_tolerance,
                    self.corridor_exit_centerline_rebase_maximum) <= 0.0 or
                self.corridor_branch_station_tolerance <= 0.0 or
                self.corridor_branch_retry_cooldown < 0.0 or
                self.corridor_branch_entry_tolerance <= 0.0 or
                self.semantic_side_door_minimum_observations < 1 or
                self.semantic_side_door_minimum_penetration <= 0.0 or
                self.corridor_side_station_search_radius < 0.0 or
                self.corridor_side_raw_frontier_minimum_size < 1 or
                self.visited_region_component_minimum_lateral <= 0.0 or
                self.visited_region_sample_spacing <= 0.0 or
                self.visited_region_maximum_station_separation <= 0.0 or
                self.region_return_relocation_radius <= 0.0 or
                self.region_exit_recovery_radius <
                self.region_return_relocation_radius or
                self.region_exit_recovery_limit < 1 or
                self.region_stall_backtrack_distance <= 0.0 or
                self.region_stall_backtrack_spacing <= 0.0 or
                self.region_stall_backtrack_limit < 1 or
                self.corridor_start_collision_backtrack_distance <= 0.0 or
                self.corridor_start_collision_backtrack_spacing <= 0.0 or
                self.corridor_start_collision_backtrack_limit < 1 or
                self.post_entry_lidar_maximum_rays < 100 or
                self.room_target_count < 1 or self.mission_home_tolerance <= 0.0 or
                self.room_phase_target_seconds <= 0.0 or
                self.room_mission_entry_minimum_remaining <= 0.0 or
                self.corridor_no_frontier_probe_after < 1 or
                self.corridor_no_frontier_probe_distance <=
                self.reached_tolerance + 0.10 or
                self.corridor_monotonic_probe_after < 1 or
                self.corridor_monotonic_probe_distance <=
                self.reached_tolerance + 0.10 or
                self.corridor_monotonic_probe_sample_spacing <= 0.0 or
                self.corridor_monotonic_probe_max_unknown_queries < 0 or
                self.corridor_monotonic_probe_limit < 1 or
                min(self.terminal_door_context_fallback_drawdown,
                    self.terminal_door_context_fallback_lateral) <= 0.0 or
                self.room_new_door_maximum_lookbehind < 0.0 or
                self.paired_opposite_maximum_backtrack < 0.0 or
                self.paired_opposite_recenter_pending_timeout <= 0.0 or
                self.room_door_maximum_corridor_alignment < 0.0 or
                self.room_door_maximum_corridor_alignment > 1.0 or
                self.local_door_maximum_corridor_alignment < 0.0 or
                self.local_door_maximum_corridor_alignment > 1.0 or
                self.room_entry_replan_wait < 0.0 or
                self.replan_grid_wait < 0.0):
            raise ValueError("invalid corridor sweep configuration")
        depth_breadth_config = DepthBreadthConfig(
            enabled=bool(rospy.get_param("~enable_depth_breadth_scheduler", False)),
            region_radius=float(rospy.get_param("~depth_breadth_region_radius", 5.0)),
            exclusion_radius=float(rospy.get_param(
                "~depth_breadth_exclusion_radius", 4.0)),
            maximum_goals=int(rospy.get_param("~depth_breadth_maximum_goals", 3)),
            maximum_path_length=float(rospy.get_param(
                "~depth_breadth_maximum_path_length", 10.0)),
            maximum_seconds=float(rospy.get_param(
                "~depth_breadth_maximum_seconds", 90.0)),
            coverage_enabled=bool(rospy.get_param(
                "~enable_region_coverage_completion", True)),
            coverage_minimum_ratio=float(rospy.get_param(
                "~region_coverage_minimum_ratio", 0.88)),
            coverage_maximum_unknown_m2=float(rospy.get_param(
                "~region_coverage_maximum_unknown_m2", 1.50)),
            coverage_minimum_known_m2=float(rospy.get_param(
                "~region_coverage_minimum_known_m2", 6.0)),
            coverage_stable_cycles=int(rospy.get_param(
                "~region_coverage_stable_cycles", 2)),
            coverage_minimum_entry_depth=float(rospy.get_param(
                "~region_coverage_minimum_entry_depth", 1.25)),
            coverage_minimum_post_entry_goals=int(rospy.get_param(
                "~region_coverage_minimum_post_entry_goals", 1)))
        room_config = LightweightRoomConfig(
            enabled=bool(rospy.get_param("~enable_lightweight_room_scheduler", False)),
            room_id_prefix=str(rospy.get_param(
                "~room_id_prefix", "estimated_room_")),
            trajectory_spacing=float(rospy.get_param("~room_trajectory_spacing", 0.20)),
            doorway_width_min=float(rospy.get_param("~room_doorway_width_min", 0.75)),
            doorway_width_max=float(rospy.get_param("~room_doorway_width_max", 2.30)),
            aperture_probe_range=float(rospy.get_param("~room_aperture_probe_range", 4.0)),
            room_depth_probe_range=float(rospy.get_param(
                "~room_depth_probe_range", 8.0)),
            room_side_probe_range=float(rospy.get_param(
                "~room_side_probe_range", 7.5)),
            expansion_margin=float(rospy.get_param("~room_expansion_margin", 0.80)),
            confirmation_depth=float(rospy.get_param("~room_confirmation_depth", 1.10)),
            minimum_crossing_depth=float(rospy.get_param(
                "~room_minimum_crossing_depth", 0.18)),
            duplicate_door_radius=float(rospy.get_param("~room_duplicate_door_radius", 1.50)),
            goal_clearance=float(rospy.get_param("~room_goal_clearance", 0.38)),
            entry_staging_clearance=float(rospy.get_param(
                "~room_entry_staging_clearance", 0.26)),
            entry_depth=float(rospy.get_param("~room_entry_depth", 1.50)),
            side_depth=float(rospy.get_param("~room_side_goal_depth", 3.00)),
            near_wall_margin=float(rospy.get_param(
                "~room_near_wall_margin", 0.65)),
            minimum_goal_separation=float(rospy.get_param(
                "~room_minimum_goal_separation", 0.90)),
            exit_offset=float(rospy.get_param("~room_exit_offset", 0.90)),
            exit_retry_limit=int(rospy.get_param("~room_exit_retry_limit", 2)),
            accept_reversed_entry_trace_exit=bool(rospy.get_param(
                "~room_accept_reversed_entry_trace_exit", False)),
            reversed_entry_trace_exit_tolerance=float(rospy.get_param(
                "~room_reversed_entry_trace_exit_tolerance_m", 0.30)),
            entry_retry_limit=int(rospy.get_param("~room_entry_retry_limit", 2)),
            candidate_confirmation_count=int(rospy.get_param(
                "~room_door_confirmation_count", 3)),
            corridor_side_confirmation_count=int(rospy.get_param(
                "~room_corridor_side_confirmation_count", 1)),
            require_portal_preflight_for_entry=bool(rospy.get_param(
                "~room_require_portal_preflight_for_entry", False)),
            door_takeover_distance=float(rospy.get_param(
                "~room_door_takeover_distance", 5.0)),
            door_lookbehind_distance=float(rospy.get_param(
                "~room_door_lookbehind_distance", 6.0)),
            door_approach_offset=float(rospy.get_param(
                "~room_door_approach_offset", 0.60)),
            room_budget_seconds=float(rospy.get_param(
                "~room_visit_budget_seconds", 75.0)),
            budget_starts_after_entry=bool(rospy.get_param(
                "~room_budget_starts_after_entry", False)),
            room_goal_timeout_seconds=float(rospy.get_param(
                "~room_goal_timeout_seconds", 28.0)),
            portal_timeout_seconds=float(rospy.get_param(
                "~room_portal_timeout_seconds", 20.0)),
            entry_timeout_seconds=float(rospy.get_param(
                "~room_entry_timeout_seconds", 45.0)),
            entry_preflight_max_path_m=float(rospy.get_param(
                "~room_entry_preflight_max_path_m", 7.5)),
            entry_preflight_max_detour_ratio=float(rospy.get_param(
                "~room_entry_preflight_max_detour_ratio", 3.0)),
            entry_candidate_astar_limit=int(rospy.get_param(
                "~room_entry_candidate_astar_limit", 40)),
            entry_trace_observed_progress_only=bool(rospy.get_param(
                "~room_entry_trace_observed_progress_only", False)),
            door_cooldown_seconds=float(rospy.get_param(
                "~room_door_cooldown_seconds", 90.0)),
            door_evidence_retry_seconds=float(rospy.get_param(
                "~room_door_evidence_retry_seconds", 3.0)),
            exit_reserve_seconds=float(rospy.get_param(
                "~room_exit_reserve_seconds", 20.0)),
            exit_progress_timeout_seconds=float(rospy.get_param(
                "~room_exit_progress_timeout_seconds", 8.0)),
            exit_timeout_seconds_per_meter=float(rospy.get_param(
                "~room_exit_timeout_seconds_per_meter", 4.0)),
            exit_timeout_overhead_seconds=float(rospy.get_param(
                "~room_exit_timeout_overhead_seconds", 8.0)),
            exit_timeout_max_seconds=float(rospy.get_param(
                "~room_exit_timeout_max_seconds", 60.0)),
            exit_anchor_skip_distance=float(rospy.get_param(
                "~room_exit_anchor_skip_distance", 0.55)),
            exit_preflight_maximum_expansions=int(rospy.get_param(
                "~room_exit_preflight_maximum_expansions", 800)),
            direct_verified_exit=bool(rospy.get_param(
                "~room_direct_verified_exit", False)),
            semantic_completion_tolerance=float(rospy.get_param(
                "~room_semantic_completion_tolerance", 0.80)),
            side_goal_retry_limit=int(rospy.get_param(
                "~room_side_goal_retry_limit", 2)),
            coverage_minimum_ratio=float(rospy.get_param(
                "~room_coverage_minimum_ratio", 0.90)),
            coverage_maximum_unknown_m2=float(rospy.get_param(
                "~room_coverage_maximum_unknown_m2", 2.00)),
            coverage_maximum_shadow_m2=float(rospy.get_param(
                "~room_coverage_maximum_shadow_m2", 1.25)),
            coverage_minimum_observations=int(rospy.get_param(
                "~room_coverage_minimum_observations", 1)),
            lidar_effective_range=float(rospy.get_param(
                "~room_lidar_effective_range", 8.0)),
            maximum_center_depth=float(rospy.get_param(
                "~room_maximum_center_depth", 3.80)),
            maximum_side_lateral=float(rospy.get_param(
                "~room_maximum_side_lateral", 3.00)),
            visual_breadth_lateral=float(rospy.get_param(
                "~room_visual_breadth_lateral_m", 0.65)),
            visual_breadth_min_baseline=float(rospy.get_param(
                "~room_visual_breadth_min_baseline_m", 1.20)),
            visual_two_pose_lateral_enabled=bool(rospy.get_param(
                "~room_visual_two_pose_full_sweep_enabled", False)),
            adaptive_minimal_viewpoints=bool(rospy.get_param(
                "~room_adaptive_minimal_viewpoints", False)),
            adaptive_maximum_viewpoints=int(rospy.get_param(
                "~room_adaptive_maximum_viewpoints", 3)),
            adaptive_prefer_central_first=bool(rospy.get_param(
                "~room_adaptive_prefer_central_first", False)),
            adaptive_occlusion_shadow_trigger_m2=float(rospy.get_param(
                "~room_adaptive_occlusion_shadow_trigger_m2", 0.55)),
            adaptive_minimum_gain_m2=float(rospy.get_param(
                "~room_adaptive_minimum_gain_m2", 0.12)),
            adaptive_path_cost_weight=float(rospy.get_param(
                "~room_adaptive_path_cost_weight", 0.32)),
            adaptive_set_cover_ratio=float(rospy.get_param(
                "~room_adaptive_set_cover_ratio", 0.88)),
            adaptive_set_cover_residual_m2=float(rospy.get_param(
                "~room_adaptive_set_cover_residual_m2", 0.75)),
            adaptive_route_return_weight=float(rospy.get_param(
                "~room_adaptive_route_return_weight", 0.65)),
            allow_single_jamb_strong_expansion=bool(rospy.get_param(
                "~room_allow_single_jamb_strong_expansion", False)),
            single_jamb_minimum_confidence=float(rospy.get_param(
                "~room_single_jamb_minimum_confidence", 0.82)),
            visited_room_overlap_min_normal_alignment=float(rospy.get_param(
                "~room_visited_overlap_min_normal_alignment", 0.25)),
            visited_room_overlap_max_door_separation=float(rospy.get_param(
                "~room_visited_overlap_max_door_separation", 2.50)))
        self.odom_topic = rospy.get_param("~odom_topic", "/Odometry")
        self.cloud_topic = rospy.get_param("~cloud_topic", "/cloud_registered")
        self.goal_topic = rospy.get_param("~goal_topic", "/exploration_goal")
        self.corridor_transit_speed = float(rospy.get_param(
            "~corridor_transit_speed", 1.20))
        self.corridor_search_speed = float(rospy.get_param(
            "~corridor_search_speed", 0.55))
        self.room_entry_speed = float(rospy.get_param(
            "~room_entry_speed", 0.38))
        self.corridor_lateral_speed_limit = float(rospy.get_param(
            "~corridor_lateral_speed_limit", 0.35))
        self.room_lateral_speed_limit = float(rospy.get_param(
            "~room_lateral_speed_limit", 0.50))
        self.room_exit_speed = float(rospy.get_param(
            "~room_exit_speed", 0.55))
        self.room_exit_lateral_speed_limit = float(rospy.get_param(
            "~room_exit_lateral_speed_limit", 0.20))
        self.return_goal_distance_gain = float(rospy.get_param(
            "~return_goal_distance_gain", 1.35))
        self.auto_generate_visualization = bool(rospy.get_param(
            "~auto_generate_visualization", True))
        self.enable_planned_room_path_intercept = bool(rospy.get_param(
            "~enable_planned_room_path_intercept", False))
        default_visualizer = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "scripts",
            "visualize_baseline_results.py"))
        self.visualization_script = os.path.abspath(rospy.get_param(
            "~visualization_script", default_visualizer))
        os.makedirs(self.output_dir, exist_ok=True)
        truth_layout = os.path.abspath(rospy.get_param(
            "~offline_truth_layout_metadata", "")) if rospy.get_param(
                "~offline_truth_layout_metadata", "") else ""
        if truth_layout and os.path.isfile(truth_layout):
            try:
                shutil.copy2(truth_layout,
                             os.path.join(self.output_dir, "layout_metadata.json"))
            except OSError as error:
                rospy.logwarn("Could not snapshot offline truth layout: %s", error)
            # Snapshot source truth beside the run for reproducible Figure 14.
            # It is copied only for post-run visualization and is never read
            # by any online navigation or RGB-D detection decision.
            truth_danger = os.path.join(os.path.dirname(truth_layout),
                                        "danger_truth.json")
            if os.path.isfile(truth_danger):
                try:
                    shutil.copy2(truth_danger,
                                 os.path.join(self.output_dir, "danger_truth.json"))
                except OSError as error:
                    rospy.logwarn("Could not snapshot offline danger truth: %s", error)
        if self.enable_scan_lite:
            try:
                open(os.path.join(self.output_dir, "scan_lite_waypoint_debug.jsonl"),
                     "w", encoding="utf-8").close()
            except OSError as error:
                rospy.logerr("Could not initialize SCAN-lite debug log: %s", error)

        self.lock = threading.RLock()
        self.process_started = time.monotonic()
        self.started = self.process_started
        self.startup_elapsed_sec = None
        self.mission_clock_started = False
        self.depth_breadth = DepthBreadthScheduler(depth_breadth_config)
        self.branch_scheduler = CorridorBranchScheduler(
            self.corridor_branch_station_tolerance,
            self.corridor_branch_retry_cooldown,
            self.corridor_branch_entry_tolerance)
        self.previous_region_was_corridor = None
        self.last_region_corridor_pose = None
        self.active_region_side_sign = None
        # After exiting a room, briefly suppress only the same-side opening.
        # Opposite-side candidates remain eligible and receive a priority
        # bonus, so corridor resumption cannot hide the other room.
        self.last_room_exit_side = None
        self.last_room_exit_pose = None
        self.room_exit_resume_until = -math.inf
        self.room_exit_map_snapshots = []
        self.room_exit_resume_distance = float(rospy.get_param(
            "~room_exit_resume_distance_m", 1.20))
        self.room_exit_resume_time = float(rospy.get_param(
            "~room_exit_resume_time_s", 5.0))
        self.room_exit_opposite_door_revisit_radius = float(rospy.get_param(
            "~room_exit_opposite_door_revisit_radius_m", 2.50))
        self.room_exit_opposite_door_revisit_time = float(rospy.get_param(
            "~room_exit_opposite_door_revisit_time_s", 15.0))
        # A confirmed opposite-side aperture can be observed just after the
        # robot has rolled past its station.  Keep its local evidence only
        # long enough to recenter on the known-free corridor-side point.
        self.pending_paired_opposite_door = None
        self.post_entry_observed_cells = set()
        self.region_exit_recovery_attempts = 0
        self.region_return_failed_targets = []
        self.region_stall_backtrack_pending = False
        self.region_stall_backtrack_attempts = 0
        self.corridor_start_collision_backtrack_attempts = 0
        self.corridor_monotonic_probe_count = 0
        self.corridor_monotonic_probe_mode_active = False
        self.visited_region_samples = []
        self.latest_cloud_xy = np.empty((0, 2), dtype=np.float32)
        self.latest_cloud_xyz = np.empty((0, 3), dtype=np.float32)
        # Map coordinates can drift by several metres during a long corridor
        # transit.  Preserve a small robot-frame LiDAR fingerprint at each
        # committed portal so that a re-observed physical doorway cannot be
        # counted as a new room merely because FAST-LIO moved its map pose.
        self.room_revisit_scan_enabled = bool(rospy.get_param(
            "~room_revisit_scan_enabled", True))
        self.room_revisit_scan_bins = int(rospy.get_param(
            "~room_revisit_scan_bins", 48))
        self.room_revisit_scan_min_bins = int(rospy.get_param(
            "~room_revisit_scan_min_bins", 16))
        self.room_revisit_scan_max_median_error = float(rospy.get_param(
            "~room_revisit_scan_max_median_error_m", 0.32))
        self.room_revisit_scan_max_width_delta = float(rospy.get_param(
            "~room_revisit_scan_max_width_delta_m", 0.35))
        self.room_revisit_scan_records = {}
        self.room_revisit_scan_log_path = os.path.join(
            self.output_dir, "logs", "room_revisit_guard.jsonl")
        self.room_scheduler = LightweightRoomScheduler(room_config)
        self.spawn_pose = {
            "x": float(rospy.get_param("~robot_spawn_x", 0.0)),
            "y": float(rospy.get_param("~robot_spawn_y", 0.8)),
            "yaw": float(rospy.get_param("~robot_spawn_yaw", 1.5708)),
            "sensor_forward_offset": float(rospy.get_param(
                "~offline_truth_sensor_forward_offset", 0.20)),
        }
        self.pose = None
        self.mission_start_pose = None
        # First online-validated G1/stair staging point for return handoff.
        self.stair_wait_anchor = None
        self.pose_frame = "camera_init"
        self.grid = None
        self.grid_update_count = 0
        self.map_statistics = {}
        self.execution_results = []
        self.locomotion_ready = False
        self.first_odom_elapsed = None
        self.first_cloud_elapsed = None
        self.map_ready_elapsed = None
        self.first_goal_elapsed = None
        self.exploration_end_elapsed = None
        self.trajectory = []
        self.goal_history = []
        self.path_history = []
        self.scan_history = []
        self.depth_breadth_history = []
        self.goal_metrics = []
        self.active_goal_id = -1
        self.active_waypoint_id = -1
        self.active_goal_distance = 0.0
        self.active_goal_last_pose = None
        self.fuel_plan_count = 0
        self.astar_success_count = 0
        self.goal_success_count = 0
        self.goal_failure_count = 0
        self.termination_reason = None
        self.state = "STARTUP"
        self.last_mission_progress_log = -math.inf
        self.map_saved_count = 0
        self.visualization_generated = False
        self.corridor_axis = None
        self.corridor_anchor = None
        self.corridor_station_axis = None
        self.corridor_station_origin = None
        self.corridor_forward_station_sign = None
        self.corridor_forward_station_high_water = None
        self.corridor_door_detection_armed = False
        self.corridor_door_arm_high_water = None
        self.corridor_door_forward_pending = False
        self.corridor_entry_forward_lock = False
        self.corridor_entry_anchor = None
        self.corridor_established_pose = None
        self.corridor_direction_aligned = False
        self.corridor_reversed = False
        self.corridor_reversal_count = 0
        self.terminal_missing_room_retrace_issued = False
        self.terminal_missing_room_no_frontier_streak = 0
        self.corridor_no_frontier_exhaustion_streak = 0
        self.terminal_missing_room_retrace_start_station = None
        # R32: 出房后 footprint 阻塞 → 探测全 occupied(地图把门口填死)
        # → NO_FRONTIER 死循环 35+ 次 → LOCAL_RESCAN 卡到 400s 超时
        # F3_EXPLORATION_FAILED。记录最近一次 forward 成功时刻,死循环
        # 无进展超阈值即触发部分返回(收尾该层探索,进入下一层/下梯)。
        self.last_corridor_forward_success_at = None
        self.corridor_confirmation_count = 0
        self.corridor_last_confirmation_grid_update = -1
        self.corridor_established = False
        self.corridor_established_from_local_door = False
        self.corridor_sweep_history = []
        # Consecutive online evidence that, after all required room exits,
        # there is no reachable forward corridor waypoint.  This complements
        # the occupied-wall test, which can be inconclusive when the terminal
        # wall is still marked unknown because of local-map drift.
        self.post_exit_forward_blocked_streak = 0
        self.local_door_status = None
        self.local_entry_status = None
        self.last_local_detector_update = -math.inf
        self.last_local_door_received = -math.inf
        self.last_local_entry_received = -math.inf
        self.local_candidate_attempts = {}
        self.initial_lobby_door_rejections = set()
        self.local_rescan_results = []
        self.last_local_rescan_completed = -math.inf
        self.last_local_rescan_requested = -math.inf
        self.local_rescan_count = 0
        self.visual_coverage_status = {}
        self.room_visual_sweep_counts = {}
        self.room_visual_two_pose_requested = set()
        self.room_visual_two_pose_completed_counts = {}
        self.lidar_visual_cue_used_rooms = set()
        self.lidar_visual_cue_log_path = os.path.join(
            self.output_dir, "logs", "lidar_visual_candidates.jsonl")
        self.room_visual_breadth_counts = {}
        self.red_ball_search_status = {}
        self.hazard_candidate_status = {}
        self.empty_visual_supplemented_rooms = set()
        # Low-confidence RGB-D candidates request one bounded standoff view.
        # They never alter room discovery or corridor direction.
        self.visual_verify_requests = []
        self.termination_decisions = []
        # FAST-LIO may keep publishing a frozen pose after registration is
        # lost.  Treat that as a mission-terminal localization fault instead
        # of repeatedly issuing in-place rescans that only contaminate the
        # saved voxel map.
        self.fastlio_registration_healthy = True
        self.fastlio_registration_invalid_count = 0
        self.fastlio_registration_lost_elapsed = None
        self.fastlio_registration_abort_invalid_count = max(
            1, int(rospy.get_param(
                "~fastlio_registration_abort_invalid_count", 3)))
        self.stair_truth_return_gate_reached = False
        self.corridor_resume_failures = 0
        self.corridor_resume_failure_limit = int(rospy.get_param(
            "~corridor_resume_failure_limit", 2))

        self.goal_pub = rospy.Publisher(self.goal_topic, PoseStamped, queue_size=1)
        self.speed_limit_pub = rospy.Publisher(
            "/simenv/goal_speed_limit", Float32, queue_size=1, latch=True)
        self.distance_gain_pub = rospy.Publisher(
            "/simenv/goal_distance_gain", Float32, queue_size=1, latch=True)
        self.lateral_speed_limit_pub = rospy.Publisher(
            "/simenv/goal_lateral_speed_limit", Float32, queue_size=1,
            latch=True)
        self.state_pub = rospy.Publisher(
            "/simenv/baseline_state", String, queue_size=1, latch=True)
        self.mission_progress_pub = rospy.Publisher(
            "/simenv/mission_progress", String, queue_size=1, latch=True)
        self.active_goal_pub = rospy.Publisher(
            "/simenv/active_goal_id", Int32, queue_size=2, latch=True)
        self.active_waypoint_pub = rospy.Publisher(
            "/simenv/active_waypoint_id", Int32, queue_size=2, latch=True)
        self.scan_duration_pub = rospy.Publisher(
            "/simenv/scan_lite_duration_ms", Float64, queue_size=5)
        self.finalize_map_pub = rospy.Publisher(
            "/simenv/finalize_voxel_map", Bool, queue_size=1, latch=True)
        self.cancel_goal_pub = rospy.Publisher(
            "/simenv/cancel_exploration_goal", String, queue_size=2)
        # Keep the return-arm event latched independently from the shared
        # state topic, which immediately advances to PLANNING/GOAL_STARTED.
        self.stair_return_transit_pub = rospy.Publisher(
            "/simenv/stair_return_transit_armed", Bool, queue_size=1,
            latch=True)
        self.local_rescan_pub = rospy.Publisher(
            "/simenv/local_rescan_request", String, queue_size=2)
        self.room_detection_enabled_pub = rospy.Publisher(
            "/simenv/room_detection_enabled", Bool, queue_size=1, latch=True)
        self.room_detection_context_pub = rospy.Publisher(
            "/simenv/room_detection_context", String, queue_size=2, latch=True)
        self.finalize_result_pub = rospy.Publisher(
            "/simenv/finalize_result", Bool, queue_size=1, latch=True)
        self.scan_service = rospy.ServiceProxy(
            "/baseline_voxel_mapper/check_twin_cylinder", CheckTwinCylinder)
        rospy.Subscriber(self.odom_topic, Odometry, self._on_odom, queue_size=30)
        rospy.Subscriber(self.cloud_topic, PointCloud2, self._on_cloud, queue_size=2)
        rospy.Subscriber("/simenv/voxel_floor_projection", OccupancyGrid,
                         self._on_grid, queue_size=2)
        rospy.Subscriber("/simenv/goal_execution_result", String,
                         self._on_execution_result, queue_size=10)
        rospy.Subscriber("/locomotion_ready", Bool,
                         self._on_locomotion_ready, queue_size=2)
        # R28: RL 策略被打断(fall-safety forcePassiveState)后 locomotion_ready
        # 永久 false,任何目标/rescan 都被 goal_executor 拒绝,探索死锁。
        # 通过 /joy 注入 FSM 命令自动恢复: RESET(10) → 固定站(1) → RL(3)。
        # 该发布器与 controller_mode_bootstrap 的 /joy 发布并存;bootstrap
        # 只在启动阶段使用,探索运行期由本发布器独占。
        self._joy_pub = rospy.Publisher("/joy", Joy, queue_size=2)
        self._fixed_stand_ready = False
        rospy.Subscriber("/fixed_stand_ready", Bool,
                         self._on_fixed_stand_ready, queue_size=2)
        self._locomotion_recovery_active = False
        # R28 恢复的 RESET 步骤(按钮10,set_model_configuration 传回出生
        # 位姿)在高层(F2/F3)是毁灭性的:第三层的机器人被传回 F1 出生点,
        # truth/里程计瞬间跳跃 → localization_jump → FASTLIO_LOCALIZATION_
        # LOST 终止(batch5 RUN2 实证,4/4 房间已完成仍被杀)。F2/F3 launch
        # 传 false:跳过 RESET,直接固定站→RL 重接。
        self._locomotion_recovery_reset = bool(rospy.get_param(
            "~locomotion_recovery_reset", True))
        # R28: rescan 被拒/无结果后调用处 continue 重入会以 0.25s 空转死循环
        # (214s 无进展)。失败计数有界,连续失败达到上限后 break 降级。
        self.failed_local_rescan_streak = 0
        self.failed_local_rescan_limit = int(rospy.get_param(
            "~local_rescan_failure_limit", 3))
        rospy.Subscriber("/simenv/voxel_map_saved", Bool,
                         self._on_map_saved, queue_size=2)
        rospy.Subscriber("/simenv/local_door_candidates", String,
                         self._on_local_door_candidates, queue_size=5)
        rospy.Subscriber("/simenv/room_entry_candidates", String,
                         self._on_room_entry_candidates, queue_size=5)
        rospy.Subscriber("/simenv/local_rescan_result", String,
                         self._on_local_rescan_result, queue_size=5)
        rospy.Subscriber("/simenv/visual_coverage", String,
                         self._on_visual_coverage, queue_size=10)
        rospy.Subscriber("/simenv/red_ball_search_status", String,
                         self._on_red_ball_search_status, queue_size=10)
        rospy.Subscriber("/simenv/hazard_candidate_status", String,
                         self._on_hazard_candidate_status, queue_size=10)
        rospy.Subscriber("/simenv/visual_verify_request", String,
                         self._on_visual_verify_request, queue_size=10)
        rospy.Subscriber("/simenv/fastlio_registration_status", String,
                         self._on_fastlio_registration_status, queue_size=5)
        rospy.Subscriber("/simenv/stair_return_gate_reached", Bool,
                         self._on_stair_truth_return_gate, queue_size=2)
        rospy.on_shutdown(self._on_shutdown)
        self.mission_progress_timer = rospy.Timer(
            rospy.Duration(5.0), self._on_mission_progress_timer)


        self.corridor_far_pair_search_target_advance = float(rospy.get_param(
            "~corridor_far_pair_search_target_advance_m", 0.0))



        self.corridor_near_pair_search_maximum_advance = max(0.0, float(
            rospy.get_param(
                "~corridor_near_pair_search_maximum_advance_m", 0.0)))
        self.corridor_near_pair_detection_wait = max(0.0, float(
            rospy.get_param(
                "~corridor_near_pair_detection_wait_seconds", 0.0)))



        forward_heading_hint = rospy.get_param(
            "~corridor_forward_heading_hint_rad", None)
        try:
            forward_heading_hint = float(forward_heading_hint)
        except (TypeError, ValueError):
            forward_heading_hint = math.nan



        self.corridor_forward_axis_hint = (
            np.asarray([math.cos(forward_heading_hint),
                        math.sin(forward_heading_hint)], dtype=float)
            if math.isfinite(forward_heading_hint) else None)



        self.corridor_entry_distance_from_mission_start = float(rospy.get_param(
            "~corridor_entry_distance_from_mission_start_m", -1.0))



        self.corridor_guided_entry_door_context = bool(rospy.get_param(
            "~corridor_guided_entry_door_context", False))
        self.corridor_guided_entry_minimum_progress = float(rospy.get_param(
            "~corridor_guided_entry_minimum_progress_m", 3.5))
        # A truth-guided upper-floor ingress is only a recovery axis, not a
        # license to skip whole doorway rows. Keep the historical fallback
        # distance unless a fresh floor instance explicitly requests short
        # observation steps (F3 does so after its map-plane reset).
        self.upper_floor_truth_probe_distance = float(rospy.get_param(
            "~upper_floor_truth_corridor_probe_distance_m", 0.0))
        self.upper_floor_truth_probe_prescan = bool(rospy.get_param(
            "~upper_floor_truth_corridor_probe_prescan", False))
        self.upper_floor_truth_probe_prescan_angle = float(rospy.get_param(
            "~upper_floor_truth_corridor_probe_prescan_angle_rad", math.pi))
        self.upper_floor_truth_probe_prescan_every = max(1, int(
            rospy.get_param(
                "~upper_floor_truth_corridor_probe_prescan_every", 1)))



        self.require_all_rooms_for_floor_handoff = bool(rospy.get_param(
            "~require_all_rooms_for_floor_handoff", True))



        self.room_phase_deadline_minimum_exited_rooms = int(rospy.get_param(
            "~room_phase_deadline_minimum_exited_rooms", 0))
        self.room_phase_deadline_hard_overrun = float(rospy.get_param(
            "~room_phase_deadline_hard_overrun_seconds", 0.0))
        self.room_phase_deadline_deferred_logged = False
        self.room_phase_active_exit_grace = float(rospy.get_param(
            "~room_phase_active_exit_grace_seconds", 45.0))
        self.room_phase_active_exit_started_at = None



        self.stop_immediately_after_all_room_exits = bool(rospy.get_param(
            "~stop_immediately_after_all_room_exits", False))



        self.corridor_terminal_past_far_station = float(rospy.get_param(
            "~corridor_terminal_past_far_station_m", 3.5))



        self.local_door_minimum_semantic_width = float(rospy.get_param(
            "~local_door_minimum_semantic_width_m", 0.75))



        self.return_room_retrace_minimum_corridor_station = float(
            rospy.get_param(
                "~return_room_retrace_minimum_corridor_station_m", -2.25))



        self.visual_sweep_settle_maximum_seconds = max(0.0, float(
            rospy.get_param("~visual_sweep_settle_maximum_seconds", 1.0)))
        self.visual_sweep_settle_hold_seconds = max(0.0, float(
            rospy.get_param("~visual_sweep_settle_hold_seconds", 0.15)))
        self.visual_sweep_settle_linear_speed = max(0.0, float(
            rospy.get_param("~visual_sweep_settle_linear_speed_mps", 0.12)))
        self.visual_sweep_settle_angular_speed = max(0.0, float(
            rospy.get_param("~visual_sweep_settle_angular_speed_rps", 0.20)))



        self.room_camera_secondary_sweep_speed = float(rospy.get_param(
            "~room_camera_secondary_sweep_angular_speed",
            self.room_camera_sweep_speed))
        self.upper_floor_far_row_door_scan_angle = float(rospy.get_param(
            "~upper_floor_far_row_door_scan_angle_rad",
            2.0 * math.pi))



        self.post_exit_opposite_entry_failures = 0
        self.post_exit_opposite_entry_failure_limit = max(1, int(
            rospy.get_param("~post_exit_opposite_entry_failure_limit", 1)))
        self.post_exit_opposite_station_holds = 0
        self.post_exit_opposite_station_hold_limit = max(1, int(
            rospy.get_param("~post_exit_opposite_station_hold_limit", 8)))
        # If the first-floor 150 s room-phase target expires while a room is
        # active, let that room finish its mandatory EXIT. Once back in the
        # corridor, keep one short observation window before latching the
        # best-effort stair return. This is when the doorway on the opposite
        # wall is most observable; immediate return skipped it in run105.
        # The window is bounded and never prevents the later stair handoff
        # when no safe opposite doorway is confirmed.
        self.room_phase_post_exit_opposite_probe_grace = max(0.0, float(
            rospy.get_param(
                "~room_phase_post_exit_opposite_probe_grace_seconds", 8.0)))
        self.last_room_exit_at = -math.inf
        self.room_phase_opposite_probe_logged_exit_at = None



        self.odom_linear_speed = math.inf
        self.odom_angular_speed = math.inf
        self.odom_speed_updated_at = -math.inf



        self.corridor_entry_elapsed_sec = None
        self.corridor_entry_source = None
        self.corridor_return_elapsed_sec = None
        self.corridor_return_reason = None
        self.corridor_return_pose = None



        self.multi_floor_handoff_pending = False
        self.external_abort_reason = None



        self.shutdown_visualization_spawned = False
        self.upper_floor_truth_probe_selected_count = 0



        self.local_confirmed_entry_memory = {}



        self.recent_exit_opposite_gave_up = set()
        # Per-doorway failure counter (keyed by "center:x,y") feeding the
        # give-up decision above; a fresh track of the same physical aperture
        # must accumulate failures, not reset them.
        self.recent_exit_opposite_failures = {}



        self.localization_truth_recovery_enabled = bool(rospy.get_param(
            "~localization_truth_recovery_enabled", True))
        self.localization_truth_recovery_grace_sec = max(
            1.0, float(rospy.get_param(
                "~localization_truth_recovery_grace_sec", 8.0)))
        self.localization_truth_recovery_max_attempts = max(
            1, int(rospy.get_param(
                "~localization_truth_recovery_max_attempts", 4)))
        self.localization_truth_recovery_start_invalid_count = max(
            1, int(rospy.get_param(
                "~localization_truth_recovery_start_invalid_count", 3)))
        self.localization_truth_recovery_retry_interval_sec = max(
            0.25, float(rospy.get_param(
                "~localization_truth_recovery_retry_interval_sec", 2.0)))
        self.localization_truth_recovery_healthy_confirmations = max(
            2, int(rospy.get_param(
                "~localization_truth_recovery_healthy_confirmations", 2)))
        self._fastlio_registration_status_sequence = 0
        self._localization_recovery_exhausted = False
        self._localization_recovery_window = LocalizationRecoveryWindow(
            self.localization_truth_recovery_grace_sec,
            self.localization_truth_recovery_retry_interval_sec,
            self.localization_truth_recovery_max_attempts,
            self.localization_truth_recovery_healthy_confirmations)
        # Compatibility mirrors are retained for diagnostics and older tests;
        # the window above is the sole state owner.
        self._localization_recovery_active = False
        self._localization_recovery_started_at = None
        self._localization_recovery_attempts = 0
        # Upper-floor truth handoff may intentionally rebase Gazebo pose while
        # FAST-LIO is still rebuilding its old map anchor.  During an active
        # truth-recovery window F1 also continues so FAST-LIO can re-anchor.
        self.allow_truth_exploration_when_registration_lost = bool(
            rospy.get_param("~allow_truth_exploration_when_registration_lost",
                            False) and
            int(rospy.get_param("~floor_number", 1)) >= 2)



        self.heading_gain_pub = rospy.Publisher(
            "/simenv/goal_heading_gain", Float32, queue_size=1, latch=True)
        self.maximum_yaw_rate_pub = rospy.Publisher(
            "/simenv/goal_maximum_yaw_rate", Float32, queue_size=1,
            latch=True)



        self.minimum_speed_pub = rospy.Publisher(
            "/simenv/goal_minimum_speed", Float32, queue_size=1, latch=True)



        self._truth_recovery_request_pub = rospy.Publisher(
            "/simenv/fastlio_truth_recovery_request", String, queue_size=2)

        self._write_all_logs()

    def elapsed(self):
        return time.monotonic() - self.started

    def _on_mission_progress_timer(self, _event):
        """Print the post-startup task clock; independent of control logic."""
        if not self.mission_clock_started or self.termination_reason is not None:
            return
        elapsed = self.elapsed()
        with self.lock:
            if time.monotonic() - self.last_mission_progress_log < 4.8:
                return
            self.last_mission_progress_log = time.monotonic()
            active_goal = self.active_goal_id
            state = self.state
        doors = list(getattr(self.room_scheduler.detector, "doors", []))
        entered = sum(bool(getattr(door, "visited", False)) for door in doors)
        exited = sum(bool(getattr(door, "completed", False)) for door in doors)
        remaining = max(0.0, self.maximum_duration - elapsed)
        payload = {
            "elapsed_sec": round(elapsed, 1),
            "maximum_duration_sec": round(self.maximum_duration, 1),
            "remaining_sec": round(remaining, 1),
            "state": state,
            "active_goal": active_goal,
            "goal_success_count": self.goal_success_count,
            "goal_failure_count": self.goal_failure_count,
            "entered_room_count": entered,
            "exited_room_count": exited,
        }
        self.mission_progress_pub.publish(
            String(data=json.dumps(payload, sort_keys=True)))
        # Keep the task clock conspicuous among the high-rate Gazebo/RL logs.
        # WARN is presentation-only; it does not indicate a mission fault.
        rospy.logwarn(
            "[MISSION TIMER] elapsed=%.1fs / %.1fs, remaining=%.1fs, "
            "state=%s, goal=%d, rooms=%d/%d",
            elapsed, self.maximum_duration, remaining, state, active_goal,
            entered, exited)

    @staticmethod
    def _atomic_json(path, payload):
        temporary = path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)

    @staticmethod
    def _load_json(path):
        try:
            with open(path, encoding="utf-8") as stream:
                return json.load(stream)
        except (OSError, ValueError):
            return None

    def _set_state(self, state, reason=None):
        self.state = state
        if self.mission_clock_started:
            payload = {"state": state, "phase": "MISSION",
                       "elapsed_sec": round(self.elapsed(), 3)}
        else:
            # Startup has a separate wall-clock diagnostic and must never be
            # mistaken for the algorithm's 200 s exploration budget.
            payload = {"state": state, "phase": "STARTUP",
                       "startup_elapsed_sec": round(self.elapsed(), 3)}
        if reason:
            payload["reason"] = reason
        self.state_pub.publish(String(data=json.dumps(payload, sort_keys=True)))
        self._publish_room_detection_context()
        self._write_all_logs()

    def _publish_room_detection_context(self):
        """Publish room-only RGB-D eligibility without affecting navigation."""
        # Red danger sources are room-interior only.  Publish a latched,
        # inspectable gate so RGB-D processing is dormant during lobby and
        # corridor transit without making the detector own navigation.
        active_door = self.room_scheduler.active_door
        room_enabled = bool(active_door is not None)
        visual_anchor = None
        if room_enabled:
            center = getattr(active_door, "center", None)
            interior = getattr(active_door, "interior_side", None)
            if center is not None and interior is not None:
                try:
                    dx, dy = float(interior[0]) - float(center[0]), float(interior[1]) - float(center[1])
                    norm = math.hypot(dx, dy)
                    if norm > 1e-4:
                        visual_anchor = {
                            "door_center": [round(float(center[0]), 4), round(float(center[1]), 4)],
                            "interior_direction": [round(dx / norm, 5), round(dy / norm, 5)],
                            "interior_yaw": round(math.atan2(dy, dx), 5),
                        }
                except (TypeError, ValueError, IndexError):
                    visual_anchor = None
        room_payload = {
            "enabled": room_enabled,
            "scheduler_state": self.room_scheduler.state,
            "room_id": (self.room_scheduler._room_id() if room_enabled else None),
            # Online doorway geometry anchors visual coverage to the room
            # side of this particular portal; no room layout or target truth
            # is supplied to the RGB-D pipeline.
            "visual_anchor": visual_anchor,
            "mission_elapsed_sec": round(self.elapsed(), 3)
            if self.mission_clock_started else None,
        }
        self.room_detection_enabled_pub.publish(Bool(data=room_enabled))
        self.room_detection_context_pub.publish(
            String(data=json.dumps(room_payload, sort_keys=True)))

    def _on_odom(self, message):
        p, q = message.pose.pose.position, message.pose.pose.orientation
        pose = (float(p.x), float(p.y), float(p.z),
                quaternion_yaw(q.x, q.y, q.z, q.w))
        if not all(math.isfinite(value) for value in pose):
            return
        linear = message.twist.twist.linear
        angular = message.twist.twist.angular
        linear_speed = math.sqrt(float(linear.x) ** 2 +
                                 float(linear.y) ** 2 + float(linear.z) ** 2)
        angular_speed = math.sqrt(float(angular.x) ** 2 +
                                  float(angular.y) ** 2 + float(angular.z) ** 2)
        with self.lock:
            self.odom_linear_speed = linear_speed
            self.odom_angular_speed = angular_speed
            self.odom_speed_updated_at = time.monotonic()
            if self.first_odom_elapsed is None:
                self.first_odom_elapsed = self.elapsed()
                self.mission_start_pose = pose
                # Upper-floor truth handoff places the robot at a verified
                # corridor ingress, but the projected map can initially have
                # no frontier or door returns.  Seed the one-way corridor
                # context at that ingress so the first bounded forward probe
                # can reach the physical room row; floor 1 remains unchanged.
                upper_floor_truth_entry = bool(
                    getattr(self, "corridor_guided_entry_door_context", None) and
                    int(rospy.get_param("~floor_number", 1)) >= 2 and
                    getattr(self, "corridor_forward_axis_hint", None) is not None)
                if upper_floor_truth_entry:
                    axis = np.asarray(self.corridor_forward_axis_hint,
                                      dtype=float)
                    axis /= max(1e-9, float(np.linalg.norm(axis)))
                    self.corridor_entry_forward_lock = True
                    self._record_corridor_entry(
                        pose[:3], "upper_floor_seeded_initial_pose",
                        elapsed_sec=0.0)
                    self.corridor_axis = axis.copy()
                    self.corridor_station_axis = axis.copy()
                    self.corridor_station_origin = np.asarray(
                        [float(pose[0]), float(pose[1])], dtype=float)
                    self.corridor_established_pose = np.asarray(
                        [float(pose[0]), float(pose[1]), float(pose[2])],
                        dtype=float)
                    self.corridor_established = True
                    self.corridor_direction_aligned = True
                    rospy.logwarn(
                        "Upper-floor truth corridor seeded at ingress "
                        "(%.2f, %.2f), axis=(%.2f, %.2f); starting bounded "
                        "forward acquisition.", pose[0], pose[1], axis[0], axis[1])
            self.pose = pose
            self.pose_frame = message.header.frame_id or self.pose_frame
            if not self.trajectory or self.elapsed() - self.trajectory[-1][0] >= 0.10:
                self.trajectory.append((self.elapsed(), pose[0], pose[1], pose[2],
                                        pose[3], self.pose_frame))
            if self.active_goal_id >= 0 and self.active_goal_last_pose is not None:
                self.active_goal_distance += math.hypot(
                    pose[0] - self.active_goal_last_pose[0],
                    pose[1] - self.active_goal_last_pose[1])
            if self.active_goal_id >= 0:
                self.active_goal_last_pose = pose
        # Odom is continuous during a long goal. It guarantees the display
        # remains live even when the simulated ROS clock pauses.
        self._on_mission_progress_timer(None)

    def _on_cloud(self, message):
        points = np.empty((0, 2), dtype=np.float32)
        points_xyz = np.empty((0, 3), dtype=np.float32)
        fields = {field.name: field for field in message.fields}
        if "x" in fields and "y" in fields and message.point_step > 0:
            endian = ">" if message.is_bigendian else "<"
            try:
                dtype = np.dtype({
                    "names": ["x", "y"],
                    "formats": [endian + "f4", endian + "f4"],
                    "offsets": [int(fields["x"].offset),
                                int(fields["y"].offset)],
                    "itemsize": int(message.point_step),
                })
                raw = np.frombuffer(message.data, dtype=dtype)
                finite = np.logical_and(
                    np.isfinite(raw["x"]), np.isfinite(raw["y"]))
                points = np.column_stack(
                    (raw["x"][finite], raw["y"][finite])).astype(
                        np.float32, copy=False)
                if "z" in fields:
                    xyz_dtype = np.dtype({
                        "names": ["x", "y", "z"],
                        "formats": [endian + "f4", endian + "f4", endian + "f4"],
                        "offsets": [int(fields["x"].offset),
                                    int(fields["y"].offset),
                                    int(fields["z"].offset)],
                        "itemsize": int(message.point_step),
                    })
                    raw_xyz = np.frombuffer(message.data, dtype=xyz_dtype)
                    finite_xyz = (np.isfinite(raw_xyz["x"]) &
                                  np.isfinite(raw_xyz["y"]) &
                                  np.isfinite(raw_xyz["z"]))
                    points_xyz = np.column_stack(
                        (raw_xyz["x"][finite_xyz], raw_xyz["y"][finite_xyz],
                         raw_xyz["z"][finite_xyz])).astype(np.float32, copy=False)
                if len(points) > self.post_entry_lidar_maximum_rays:
                    stride = int(math.ceil(
                        len(points) / float(self.post_entry_lidar_maximum_rays)))
                    points = points[::stride]
                if len(points_xyz) > self.post_entry_lidar_maximum_rays:
                    stride = int(math.ceil(len(points_xyz) / float(
                        self.post_entry_lidar_maximum_rays)))
                    points_xyz = points_xyz[::stride]
            except (TypeError, ValueError):
                points = np.empty((0, 2), dtype=np.float32)
                points_xyz = np.empty((0, 3), dtype=np.float32)
        with self.lock:
            if self.first_cloud_elapsed is None:
                self.first_cloud_elapsed = self.elapsed()
            self.latest_cloud_xy = points
            self.latest_cloud_xyz = points_xyz

    def _on_grid(self, message):
        try:
            data = np.asarray(message.data, dtype=np.int16).reshape(
                message.info.height, message.info.width)
            grid = OccupancyGrid2D(data, float(message.info.resolution),
                                   float(message.info.origin.position.x),
                                   float(message.info.origin.position.y))
        except (TypeError, ValueError):
            return
        with self.lock:
            self.grid = grid
            self.grid_update_count += 1

    def _lidar_visual_cue(self, room_id):
        """Return one compact in-room LiDAR object cue for an RGB-D fan.

        This deliberately classifies neither balls nor hazards.  It keeps
        only small, isolated 3-D return components on the interior side of
        the active doorway, then supplies their bearing to the camera sweep.
        Red/round/depth validation and multi-frame confirmation remain in the
        RGB-D detector/tracker.
        """
        if (not self.lidar_visual_cue_enabled or not room_id or
                room_id in self.lidar_visual_cue_used_rooms or
                not self.depth_breadth.entry_observed):
            return None
        with self.lock:
            pose = self.pose
            cloud = np.asarray(self.latest_cloud_xyz, dtype=np.float32).copy()
        door = self.room_scheduler.active_door
        if pose is None or door is None or len(cloud) < 3:
            return None
        center = np.asarray(door.center, dtype=float)
        inward = np.asarray(door.normal, dtype=float)
        inward_norm = float(np.linalg.norm(inward))
        if inward_norm < 1e-6:
            return None
        inward /= inward_norm
        lateral = np.asarray([-inward[1], inward[0]])
        dx = cloud[:, 0] - float(pose[0])
        dy = cloud[:, 1] - float(pose[1])
        ranges = np.hypot(dx, dy)
        from_door_x = cloud[:, 0] - center[0]
        from_door_y = cloud[:, 1] - center[1]
        interior_depth = from_door_x * inward[0] + from_door_y * inward[1]
        interior_lateral = np.abs(from_door_x * lateral[0] + from_door_y * lateral[1])
        mask = ((ranges >= self.lidar_visual_cue_min_range) &
                (ranges <= self.lidar_visual_cue_max_range) &
                (cloud[:, 2] >= self.lidar_visual_cue_height_min) &
                (cloud[:, 2] <= self.lidar_visual_cue_height_max) &
                (interior_depth >= .20) & (interior_depth <= 6.5) &
                (interior_lateral <= 5.0))
        filtered = cloud[mask]
        if len(filtered) < 3:
            return None
        cell_size = max(.10, self.lidar_visual_cue_cell_size)
        cells = {}
        for index, point in enumerate(filtered):
            cell = (int(math.floor(float(point[0]) / cell_size)),
                    int(math.floor(float(point[1]) / cell_size)))
            cells.setdefault(cell, []).append(index)
        visited, components = set(), []
        for start in cells:
            if start in visited:
                continue
            stack, indices = [start], []
            visited.add(start)
            while stack:
                cell = stack.pop()
                indices.extend(cells[cell])
                for ox in (-1, 0, 1):
                    for oy in (-1, 0, 1):
                        neighbour = (cell[0] + ox, cell[1] + oy)
                        if neighbour in cells and neighbour not in visited:
                            visited.add(neighbour)
                            stack.append(neighbour)
            if 1 <= len(indices) and len(indices) <= self.lidar_visual_cue_max_cells * 8:
                component_cells = {(
                    int(math.floor(float(filtered[i, 0]) / cell_size)),
                    int(math.floor(float(filtered[i, 1]) / cell_size)))
                    for i in indices}
                if len(component_cells) <= self.lidar_visual_cue_max_cells:
                    components.append((indices, component_cells))
        candidates = []
        for indices, component_cells in components:
            cluster = filtered[indices]
            centroid = np.mean(cluster, axis=0)
            radius = math.hypot(float(centroid[0]) - pose[0],
                                float(centroid[1]) - pose[1])
            bearing = ((math.atan2(float(centroid[1]) - pose[1],
                                    float(centroid[0]) - pose[0]) - pose[3] + math.pi) %
                       (2.0 * math.pi) - math.pi)
            # Compactness rejects walls/furniture faces.  A component with a
            # few dense returns remains eligible, since a small sphere is
            # sparse at long range in this simulator.
            span = (float(np.max(cluster[:, :2], axis=0)[0] -
                          np.min(cluster[:, :2], axis=0)[0]),
                    float(np.max(cluster[:, :2], axis=0)[1] -
                          np.min(cluster[:, :2], axis=0)[1]))
            if max(span) > .65:
                continue
            candidates.append({
                "position": [round(float(centroid[0]), 3),
                             round(float(centroid[1]), 3),
                             round(float(centroid[2]), 3)],
                "range_m": round(radius, 3),
                "bearing_rad": round(bearing, 3),
                "component_cells": len(component_cells),
                "point_count": len(indices),
                "span_m": [round(span[0], 3), round(span[1], 3)],
            })
        if not candidates:
            return None
        # Prefer a centred, nearby compact object: it needs the least turn
        # and has the best RGB-D depth reliability.
        best = min(candidates, key=lambda item: (
            abs(item["bearing_rad"]), item["range_m"], item["component_cells"]))
        best["room_id"] = room_id
        best["candidate_count"] = len(candidates)
        best["elapsed_sec"] = round(self.elapsed(), 3)
        try:
            with open(self.lidar_visual_cue_log_path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps({"event": "LIDAR_VISUAL_CUE_SELECTED",
                                         "selected": best,
                                         "candidates": candidates},
                                        sort_keys=True) + "\n")
        except OSError:
            pass
        self.lidar_visual_cue_used_rooms.add(room_id)
        return best

    @staticmethod
    def _decode_json_message(message):
        try:
            value = json.loads(message.data)
            return value if isinstance(value, dict) else None
        except (TypeError, ValueError):
            return None

    def _on_local_door_candidates(self, message):
        payload = self._decode_json_message(message)
        if payload is None:
            return
        with self.lock:
            self.local_door_status = payload
            self.last_local_detector_update = self.elapsed()
            if int(payload.get("side_opening_count", 0)) > 0:
                self.last_local_door_received = self.elapsed()

    def _on_room_entry_candidates(self, message):
        payload = self._decode_json_message(message)
        if payload is None:
            return
        with self.lock:
            self.local_entry_status = payload
            if int(payload.get("room_entry_candidate_count", 0)) > 0:
                self.last_local_entry_received = self.elapsed()

    def _on_local_rescan_result(self, message):
        payload = self._decode_json_message(message)
        if payload is None:
            return
        payload["received_elapsed_sec"] = round(self.elapsed(), 3)
        with self.lock:
            self.local_rescan_results.append(payload)
            if payload.get("success"):
                self.last_local_rescan_completed = self.elapsed()

    def _on_visual_coverage(self, message):
        payload = self._decode_json_message(message)
        room_id = payload.get("room_id") if isinstance(payload, dict) else None
        if room_id:
            with self.lock:
                self.visual_coverage_status[str(room_id)] = payload

    def _on_red_ball_search_status(self, message):
        payload = self._decode_json_message(message)
        room_id = payload.get("room_id") if isinstance(payload, dict) else None
        if room_id:
            with self.lock:
                self.red_ball_search_status[str(room_id)] = payload

    def _on_hazard_candidate_status(self, message):
        payload = self._decode_json_message(message)
        room_id = payload.get("room_id") if isinstance(payload, dict) else None
        if room_id:
            with self.lock:
                self.hazard_candidate_status[str(room_id)] = payload

    def _on_visual_verify_request(self, message):
        payload = self._decode_json_message(message)
        position = payload.get("position", {}) if isinstance(payload, dict) else {}
        try:
            x, y = float(position["x"]), float(position["y"])
        except (KeyError, TypeError, ValueError):
            return
        payload["position"] = {"x": x, "y": y}
        payload["received_elapsed_sec"] = self.elapsed()
        with self.lock:
            # Coalesce repeated observations of the same unconfirmed track.
            candidate_id = payload.get("candidate_id")
            self.visual_verify_requests = [item for item in self.visual_verify_requests
                                           if item.get("candidate_id") != candidate_id]
            self.visual_verify_requests.append(payload)

    def _plan_visual_verify_goal(self):
        """Return one A*/SCAN-lite standoff goal, or None when inappropriate."""
        with self.lock:
            pose = self.pose
            requests = list(self.visual_verify_requests)
            self.visual_verify_requests = []
        if pose is None or not requests or not self.depth_breadth.entry_observed:
            return None
        request = requests[-1]
        target = request["position"]
        dx, dy = target["x"] - pose[0], target["y"] - pose[1]
        distance = math.hypot(dx, dy)
        if distance < 1.0 or distance > 7.0:
            return None
        # Stop 2m short of the candidate, which lies inside the requested
        # 1.5--3m verification band.  The normal _plan_path pipeline performs
        # occupancy A* then SCAN-lite footprint validation before execution.
        standoff = 2.0
        scale = max(0.0, (distance - standoff) / distance)
        point = [pose[0] + dx * scale, pose[1] + dy * scale, math.atan2(dy, dx)]
        return {"position": point, "source": "lightweight_room_visual_verify",
                "room_role": "VISUAL_VERIFY", "room_id": request.get("room_id"),
                "candidate_id": request.get("candidate_id"),
                "planning_clearance_m": self.clearance}

    def _accumulate_post_entry_lidar(self, grid, pose):
        """Rasterize only LiDAR rays emitted while physically inside a branch."""
        if (not self.depth_breadth.entry_observed or
                self.depth_breadth.anchor is None):
            return
        with self.lock:
            points = np.asarray(self.latest_cloud_xy, dtype=np.float32).copy()
        start = grid.world_to_cell((float(pose[0]), float(pose[1])))
        if start is None or not len(points):
            return
        center = self.depth_breadth.anchor
        radius = self.depth_breadth.config.region_radius
        radius_sq = radius * radius

        def ray_cells(a, b):
            x0, y0 = a
            x1, y1 = b
            dx, dy = abs(x1 - x0), abs(y1 - y0)
            sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
            error = dx - dy
            while True:
                yield x0, y0
                if x0 == x1 and y0 == y1:
                    break
                twice = 2 * error
                if twice > -dy:
                    error -= dy
                    x0 += sx
                if twice < dx:
                    error += dx
                    y0 += sy

        for endpoint in points:
            if math.hypot(
                    float(endpoint[0]) - float(pose[0]),
                    float(endpoint[1]) - float(pose[1])) > 30.5:
                continue
            end = grid.world_to_cell(endpoint)
            if end is None:
                continue
            for cell in ray_cells(start, end):
                if not (0 <= cell[0] < grid.width and
                        0 <= cell[1] < grid.height):
                    break
                world = grid.cell_to_world(cell)
                if ((world[0] - center[0]) ** 2 +
                        (world[1] - center[1]) ** 2 > radius_sq):
                    continue
                state = int(grid.data[cell[1], cell[0]])
                if state >= 50:
                    break
                self.post_entry_observed_cells.add(cell)

    def _corridor_station(self, point):
        if (self.corridor_station_axis is None or
                self.corridor_station_origin is None):
            return 0.0
        axis = np.asarray(self.corridor_station_axis, dtype=float)
        origin = np.asarray(self.corridor_station_origin, dtype=float)
        return float(np.dot(np.asarray(point[:2], dtype=float) - origin, axis))

    def _synchronize_corridor_station_frame_after_relock(
            self, relocked_axis, centerline_point, pose):
        """Keep the immutable station frame aligned with a pre-room relock."""
        axis = np.asarray(relocked_axis, dtype=float)
        axis /= max(1e-9, float(np.linalg.norm(axis)))
        center = np.asarray(centerline_point[:2], dtype=float)
        old_axis = (None if self.corridor_station_axis is None else
                    np.asarray(self.corridor_station_axis, dtype=float))
        old_origin = (None if self.corridor_station_origin is None else
                      np.asarray(self.corridor_station_origin, dtype=float))
        new_origin = (center.copy() if old_origin is None else
                      np.asarray(rebase_corridor_line_laterally(
                          old_origin, axis, center), dtype=float))
        self.corridor_axis = axis.copy()
        self.corridor_anchor = center.copy()
        self.corridor_station_axis = axis.copy()
        self.corridor_station_origin = new_origin
        self.corridor_forward_station_sign = 1.0
        current = np.asarray(pose[:2], dtype=float)
        self.corridor_forward_station_high_water = float(
            np.dot(current - new_origin, axis))
        self.corridor_station_centerline_rebased = False
        self.corridor_station_centerline_calibrated = True
        return {
            "old_axis": (old_axis.tolist() if old_axis is not None else None),
            "new_axis": axis.tolist(),
            "old_origin": (old_origin.tolist()
                           if old_origin is not None else None),
            "new_origin": new_origin.tolist(),
        }

    def _door_centerline_lateral_admissible(self, door_center, source,
                                             candidate_id=None):
        """Reject an aperture inside the stable corridor centre band.

        This uses only the online corridor station frame.  It intentionally
        has no floor/layout truth dependency and is disabled by default; F2
        and later opt in through their fresh process-local parameters.
        """
        minimum = float(getattr(
            self, "room_door_minimum_centerline_lateral", 0.0))
        if minimum <= 0.0:
            return True
        if (self.corridor_station_axis is None or
                self.corridor_station_origin is None):
            return True
        axis = np.asarray(self.corridor_station_axis, dtype=float)
        norm = float(np.linalg.norm(axis))
        if norm < 1e-6:
            return True
        axis /= norm
        normal = np.asarray([-axis[1], axis[0]], dtype=float)
        origin = np.asarray(self.corridor_station_origin, dtype=float)
        center = np.asarray(door_center[:2], dtype=float)
        lateral = abs(float(np.dot(center - origin, normal)))
        if self._post_exit_opposite_centerline_exception(center, lateral):
            self.corridor_sweep_history.append({
                "event": "POST_EXIT_OPPOSITE_CENTERLINE_BAND_RELAXED",
                "elapsed_sec": round(self.elapsed(), 3),
                "source": str(source),
                "candidate_id": candidate_id,
                "candidate_center": [float(center[0]), float(center[1])],
                "centerline_lateral_m": round(lateral, 3),
                "minimum_centerline_lateral_m": minimum,
                "policy": "same_station_opposite_after_physical_exit",
            })
            return True
        if lateral >= minimum:
            return True
        self.corridor_sweep_history.append({
            "event": "DOOR_REJECTED_INSIDE_CORRIDOR_CENTER_BAND",
            "elapsed_sec": round(self.elapsed(), 3),
            "source": str(source),
            "candidate_id": candidate_id,
            "candidate_center": [float(center[0]), float(center[1])],
            "centerline_lateral_m": round(lateral, 3),
            "minimum_centerline_lateral_m": minimum,
        })
        return False

    def _terminal_missing_room_retrace_progress_met(self, pose):
        """Gate a reverse observation pass by online outbound progress."""
        required = max(
            0.0,
            float(self.terminal_missing_room_retrace_minimum_outbound_progress))
        if required <= 0.0:
            return True
        if (pose is None or self.corridor_station_axis is None or
                self.corridor_station_origin is None):
            return False
        sign = (self.corridor_forward_station_sign
                if self.corridor_forward_station_sign is not None else 1.0)
        return bool(float(sign) * self._corridor_station(pose) >= required)

    def _partial_corridor_return_trigger(self):
        """Return a timed or progress-based best-effort return trigger."""
        room_phase_started = getattr(
            getattr(self, "room_scheduler", None),
            "first_room_started_at", None)
        if (bool(getattr(self, "enforce_room_phase_target_deadline", False)) and
                room_phase_started is not None and
                self.elapsed() - float(room_phase_started) >= float(getattr(
                    self, "room_phase_target_seconds", math.inf))):
            return "room_phase_deadline"
        required = max(0.0, float(getattr(
            self, "corridor_partial_return_minimum_outbound_progress", 0.0)))
        high_water = getattr(self, "corridor_forward_station_high_water", None)
        if (required > 0.0 and high_water is not None and
                float(high_water) >= required - 0.25):
            return "online_outbound_progress"
        reserve = max(0.0, float(getattr(
            self, "corridor_partial_return_reserve", 0.0)))
        if reserve > 0.0:
            remaining = max(0.0, float(self.maximum_duration) - self.elapsed())
            if remaining <= reserve:
                return "next_floor_time_reserve"
        # R32: 出房后 start_footprint_blocked → 探测样本全 occupied(地图
        # 把门口填死)→ NO_FRONTIER 死循环 35+ 次、LOCAL_RESCAN 卡到
        # 400s 超时。forward/short/monotonic probe 全失败且无进展超过
        # 阈值即收尾返回(该层已尽力,进入下一层/下梯)。
        stuck_streak = max(0, int(getattr(
            self, "corridor_no_frontier_return_streak", 5)))
        stuck_sec = float(getattr(
            self, "corridor_no_frontier_return_after_sec", 60.0))
        last_success = self.last_corridor_forward_success_at
        since_success = (self.elapsed() if last_success is None else
                         self.elapsed() - float(last_success))
        if (self.corridor_no_frontier_exhaustion_streak >= stuck_streak and
                since_success >= stuck_sec):
            return "no_frontier_exhaustion"
        return None


    def _corridor_station_entry_origin(self, fallback):
        """Return the online narrow-corridor entrance for station gating.

        Corridor geometry can become confirmed beside the first room pair.
        Using that late pose as station zero makes the real doors look like
        lobby-throat openings.  Conversely, the earlier stair-wait anchor is
        still inside the broad lobby and made run24 count the lobby's north
        side as rooms 0/1.  The forward-lock pose is the first online pose
        that satisfies the narrow, elongated corridor geometry, so it lies
        between those two failure modes without consulting map truth.
        """
        if self.corridor_entry_anchor is not None:
            return np.asarray(
                self.corridor_entry_anchor[:2], dtype=float).copy()
        if self.stair_wait_anchor is not None:
            return np.asarray(self.stair_wait_anchor[:2], dtype=float).copy()
        return np.asarray(fallback, dtype=float).copy()

    def _reject_initial_lobby_throat(self, door_center, candidate_id,
                                     source):
        """Reject lobby/corridor transitions before the first real room.

        Corridor establishment defines an online station origin after the
        robot has crossed the lobby throat.  A lateral opening behind, or only
        just beyond, that origin is therefore part of the entrance geometry,
        not a room farther down the corridor.  Apply this only until one room
        has been physically entered so later paired-door handling is unchanged.
        """
        if (self.corridor_station_axis is None or
                self.corridor_station_origin is None):
            return False
        sign = (self.corridor_forward_station_sign
                if self.corridor_forward_station_sign is not None else 1.0)
        directed_station = float(sign) * self._corridor_station(door_center)
        # The terminal retrace is allowed to recover missed side doors, but
        # only inside the already established corridor station interval.  In
        # run25 it passed G2, promoted lobby openings at stations -2 m and
        # -4 m as rooms, and eventually became trapped in a fake room exit.
        # This bound is online: the station origin was latched when corridor
        # geometry first became established, not read from layout truth.
        return_pass_lobby = bool(
            self._missing_room_retrace_active() and
            directed_station < self.first_room_minimum_corridor_station)
        already_entered_room = any(
            bool(door.visited) for door in self.room_scheduler.detector.doors)
        if already_entered_room and not return_pass_lobby:
            return False
        if directed_station >= self.first_room_minimum_corridor_station:
            return False
        identifier = str(candidate_id or "unidentified-door")
        log_key = (str(source), identifier)
        if log_key not in self.initial_lobby_door_rejections:
            self.initial_lobby_door_rejections.add(log_key)
            self.corridor_sweep_history.append({
                "event": ("RETURN_ROOM_CANDIDATE_REJECTED_BEYOND_G2"
                          if return_pass_lobby else
                          "INITIAL_ROOM_CANDIDATE_REJECTED_LOBBY_THROAT"),
                "elapsed_sec": round(self.elapsed(), 3),
                "candidate_id": identifier,
                "door_center": [float(door_center[0]),
                                float(door_center[1])],
                "directed_station_m": round(directed_station, 3),
                "minimum_station_m":
                    self.first_room_minimum_corridor_station,
                "source": str(source),
            })
        return True

    def _update_corridor_forward_high_water(self, pose, context):
        """Track forward corridor progress without a truth/global threshold."""
        if (pose is None or context is None or
                (not context.get("terminal_axis_fallback", False) and
                 not self._on_established_corridor_centerline(pose, context)) or
                self.corridor_station_axis is None):
            return None
        station = self._corridor_station(pose)
        if self.corridor_forward_station_sign is None:
            # ``corridor_station_axis`` is seeded from the already oriented
            # corridor axis (away from the mission start).  Inferring another
            # sign from the first non-zero station is unstable after a room
            # exit: the first pose can lie on the opposite side of the saved
            # anchor and permanently label all later forward doors as
            # "behind progress".  Use the station frame's defined forward
            # direction directly.
            self.corridor_forward_station_sign = 1.0
        directed = float(self.corridor_forward_station_sign) * station
        if self.corridor_forward_station_high_water is None:
            self.corridor_forward_station_high_water = directed
        else:
            self.corridor_forward_station_high_water = max(
                self.corridor_forward_station_high_water, directed)
        return {
            "station": directed,
            "high_water": self.corridor_forward_station_high_water,
            "drawdown": self.corridor_forward_station_high_water - directed,
        }

    def _at_corridor_forward_high_water(self, pose, context):
        # The high-water gate prevents an ordinary outbound sweep from being
        # stolen by stale apertures behind the robot.  A terminal retrace is
        # different: it exists specifically to inspect those earlier corridor
        # stations again.  Keeping the outbound high-water requirement active
        # during that pass made every side-door planner return before invoking
        # its detector (DOOR_TAKEOVER_REJECTED_DURING_BACKTRACK).
        if (self._missing_room_retrace_active() or
                self._terminal_return_room_supplement_active()):
            return True
        progress = self._update_corridor_forward_high_water(pose, context)
        return bool(progress is not None and
                    progress["drawdown"] <=
                    self.local_door_forward_high_water_tolerance)

    def _door_search_corridor_context(self, pose, grid, source):
        """Return live corridor geometry or a bounded terminal-axis fallback.

        Near an end wall the rolling window is no longer elongated enough to
        satisfy the generic corridor classifier.  That must not hide the last
        side-door pair.  The fallback uses only the previously confirmed
        online axis and current pose and is admitted close to the outbound
        high-water station and centreline.  Door admission remains subject to
        all normal local sensing and path checks.
        """
        context = self._corridor_context(pose, grid)
        if self._on_established_corridor_centerline(pose, context):
            # The generic established-centreline test intentionally tolerates
            # an open doorway and can therefore remain true in a drifted room
            # whose map pose lies near the old corridor line.  When the
            # terminal fallback is disabled by F2, require current raw wall
            # geometry here as well; otherwise the early return would bypass
            # the floor-specific guard below.
            if (self.enable_terminal_door_context_fallback or
                    bool(context.get("raw_is_corridor"))):
                return context
            return None
        if not self.enable_terminal_door_context_fallback:
            return None
        if (pose is None or self.corridor_station_axis is None or
                self.corridor_station_origin is None or
                self.corridor_forward_station_high_water is None or
                self.corridor_forward_station_sign is None):
            return None
        axis = np.asarray(self.corridor_station_axis, dtype=float)
        norm = float(np.linalg.norm(axis))
        if norm < 1e-6:
            return None
        axis /= norm
        origin = np.asarray(self.corridor_station_origin, dtype=float)
        current = np.asarray([float(pose[0]), float(pose[1])], dtype=float)
        longitudinal = float(np.dot(current - origin, axis))
        centre = origin + longitudinal * axis
        lateral = float(np.linalg.norm(current - centre))
        directed_station = (float(self.corridor_forward_station_sign) *
                            longitudinal)
        drawdown = (float(self.corridor_forward_station_high_water) -
                    directed_station)
        if (drawdown < -0.35 or
                drawdown > self.terminal_door_context_fallback_drawdown or
                lateral > self.terminal_door_context_fallback_lateral):
            return None
        fallback = {
            "axis": [float(axis[0]), float(axis[1])],
            "heading": math.atan2(float(axis[1]), float(axis[0])),
            "centerline_point": [float(centre[0]), float(centre[1])],
            "is_corridor": True,
            "raw_is_corridor": False,
            "terminal_axis_fallback": True,
        }
        event_key = ("terminal-door-context", str(source))
        if event_key not in self.initial_lobby_door_rejections:
            self.initial_lobby_door_rejections.add(event_key)
            self.corridor_sweep_history.append({
                "event": "TERMINAL_DOOR_CONTEXT_FALLBACK_ENABLED",
                "elapsed_sec": round(self.elapsed(), 3),
                "source": str(source),
                "drawdown_m": round(drawdown, 3),
                "lateral_error_m": round(lateral, 3),
                "policy": "confirmed_online_axis_plus_local_door_checks",
            })
        return fallback

    def _room_exit_corridor_confirmed(self, pose, grid, goal=None):
        """Verify that a successful EXIT ended in the physical corridor.

        The station-band and raw-raster guards are independently opt-in.  The
        former catches an exit that remains laterally inside a room without
        penalising a real open doorway; the latter is available where full
        bilateral corridor geometry is known to remain observable.
        """
        require_raw = bool(getattr(
            self, "room_exit_require_raw_corridor_confirmation", False))
        require_station = bool(getattr(
            self, "room_exit_require_station_corridor_confirmation", False))
        if not (require_raw or require_station):
            return True, None
        if pose is None or (require_raw and grid is None):
            return False, "exit_corridor_map_or_pose_unavailable"
        # Check the immutable station line before rebuilding local corridor
        # geometry.  A legitimate EXIT immediately beside an open doorway can
        # make one lateral ray pass through the room and fail raw corridor
        # width classification even though the robot is on the centreline.
        if require_station:
            default_station_tolerance = float(getattr(
                self, "corridor_centerline_membership_tolerance", 0.75))
            exit_station_tolerance = float(getattr(
                self, "room_exit_station_corridor_tolerance",
                default_station_tolerance))
            station_confirmed = self._on_station_corridor_centerline(
                pose, tolerance=exit_station_tolerance)
            if not station_confirmed:
                target = ((goal or {}).get("position") or [])
                door = self.room_scheduler.active_door
                target_in_band = bool(
                    len(target) >= 2 and
                    self._on_station_corridor_centerline(
                        target, tolerance=exit_station_tolerance))
                door_depth = (door.depth((float(pose[0]), float(pose[1])))
                              if door is not None else math.inf)
                endpoint_tolerance = min(
                    0.30, float(self.room_scheduler.config.
                                semantic_completion_tolerance))
                endpoint_accepted = room_exit_endpoint_tolerance_allowed(
                    pose, target, door_depth, target_in_band,
                    endpoint_tolerance=endpoint_tolerance)
                if not endpoint_accepted:
                    return False, "exit_outside_established_corridor_band"
                self.corridor_sweep_history.append({
                    "event": "ROOM_EXIT_STATION_ENDPOINT_TOLERANCE_ACCEPTED",
                    "elapsed_sec": round(self.elapsed(), 3),
                    "room_id": (goal or {}).get("room_id"),
                    "final_pose": [float(pose[0]), float(pose[1])],
                    "commanded_endpoint": [float(target[0]), float(target[1])],
                    "endpoint_error_m": round(math.hypot(
                        float(pose[0]) - float(target[0]),
                        float(pose[1]) - float(target[1])), 3),
                    "door_plane_depth_m": round(float(door_depth), 3),
                    "policy": "in_band_target_plus_corridor_side_crossing",
                })
        if require_raw:
            context = self._corridor_context(pose, grid)
            if context is None or not bool(context.get("raw_is_corridor")):
                return False, "exit_raw_corridor_geometry_unconfirmed"
        return True, None

    def _missing_room_retrace_active(self):
        """Whether the constrained reverse pass may recover unvisited rooms."""
        return bool(
            self.enable_terminal_missing_room_retrace and
            self.terminal_missing_room_retrace_issued and
            self.corridor_reversed and
            not self._mission_rooms_physically_exited())

    def _record_missing_room_retrace_start(self, pose):
        """Remember the fixed-frame station where a reverse observation starts."""
        if (pose is None or self.corridor_station_axis is None or
                self.corridor_station_origin is None):
            self.terminal_missing_room_retrace_start_station = None
            return
        sign = (self.corridor_forward_station_sign
                if self.corridor_forward_station_sign is not None else 1.0)
        self.terminal_missing_room_retrace_start_station = (
            float(sign) * self._corridor_station(pose))

    def _finish_bounded_missing_room_retrace(self, pose):
        """Resume outbound search once the reverse observation window is spent."""
        if (not self._missing_room_retrace_active() or pose is None or
                self.terminal_missing_room_retrace_start_station is None or
                self.corridor_station_axis is None):
            return False
        sign = (self.corridor_forward_station_sign
                if self.corridor_forward_station_sign is not None else 1.0)
        directed = float(sign) * self._corridor_station(pose)
        travelled = max(
            0.0,
            float(self.terminal_missing_room_retrace_start_station) - directed)
        if travelled < self.terminal_missing_room_retrace_max_distance:
            return False
        outbound = np.asarray(self.corridor_station_axis, dtype=float)
        norm = float(np.linalg.norm(outbound))
        if norm < 1e-6:
            return False
        outbound /= norm
        if float(sign) < 0.0:
            outbound = -outbound
        self.corridor_axis = outbound
        self.corridor_reversed = False
        # A bounded false-terminal recovery may be followed by one genuine
        # terminal pass.  The independent reversal limit prevents cycling.
        self.terminal_missing_room_retrace_issued = False
        self.terminal_missing_room_no_frontier_streak = 0
        self.terminal_missing_room_retrace_start_station = None
        self.corridor_sweep_history.append({
            "event":
                "TERMINAL_MISSING_ROOM_RETRACE_BOUND_REACHED_RESUME_OUTBOUND",
            "elapsed_sec": round(self.elapsed(), 3),
            "travelled_m": round(travelled, 3),
            "limit_m": round(
                self.terminal_missing_room_retrace_max_distance, 3),
            "axis": [float(outbound[0]), float(outbound[1])],
        })
        return True

    def _new_door_is_behind_corridor_progress(self, door_center, pose):
        """Reject a newly observed door that would make corridor search reverse.

        A local detector can continue to see an aperture for several metres
        after the robot has passed it.  That is useful while finishing an
        already committed room, but it must not preempt forward corridor
        discovery with a new behind-the-robot room entry.
        """
        if (self.corridor_station_axis is None or
                self.corridor_station_origin is None or pose is None):
            return False, 0.0
        sign = (self.corridor_forward_station_sign
                if self.corridor_forward_station_sign is not None else 1.0)
        door_station = self._corridor_station(door_center)
        pose_station = self._corridor_station(pose)
        directed_delta = float(sign) * (door_station - pose_station)
        return (directed_delta < -self.room_new_door_maximum_lookbehind,
                directed_delta)

    def _local_door_has_room_area(self, candidate):
        """Accept unseen space or a large LiDAR-visible free interior."""
        return bool(
            float(candidate.get("unknown_behind_m2", 0.0)) >=
            self.local_door_minimum_unknown or
            float(candidate.get("free_behind_m2", 0.0)) >=
            self.local_door_minimum_free)

    def _update_corridor_door_phase(self, pose, context):
        """Latch room/door processing only after sustained corridor travel.

        Before this latch the exploration stack remains in its original
        lobby-egress/SEEK_CORRIDOR behaviour: FUEL and the corridor forward
        modifier may move the robot, but no side opening can take control.
        The progress is relative to the online corridor origin and never uses
        a truth-room coordinate or layout metadata.
        """
        if self.corridor_door_detection_armed:
            return True
        # A local doorway candidate is also valid corridor evidence.  In the
        # failed opt12 run the detector produced confirmed, reachable side
        # openings, but the rolling corridor confirmation counter was reset by
        # a single noisy map update.  That left corridor_established=false and
        # made the room scheduler discard every real door.  Once the robot has
        # made a modest forward displacement, promote a strong local opening
        # together with raw corridor geometry to the corridor phase.  This is
        # entirely online (no truth/layout coordinates) and still prevents
        # lobby openings at the spawn pose from taking over.
        if not self.corridor_established:
            # A lobby opening must never create a corridor phase by itself.
            # A local doorway normally assists while the current occupancy
            # window is corridor-shaped.  At a real side aperture one wall
            # return can disappear, however, so also accept the already
            # proven corridor-entry lock after a bounded forward advance on
            # its line.  The lock is acquired only after leaving the lobby.
            locked_progress = 0.0
            locked_lateral_error = math.inf
            if (self.corridor_entry_forward_lock and
                    self.corridor_entry_anchor is not None and pose is not None):
                locked_axis = np.asarray(
                    context.get("axis", [1.0, 0.0]), dtype=float)
                locked_axis /= max(
                    1e-9, float(np.linalg.norm(locked_axis)))
                locked_delta = (
                    np.asarray([float(pose[0]), float(pose[1])], dtype=float) -
                    np.asarray(self.corridor_entry_anchor[:2], dtype=float))
                locked_progress = float(np.dot(locked_delta, locked_axis))
                locked_lateral_error = abs(float(
                    locked_axis[0] * locked_delta[1] -
                    locked_axis[1] * locked_delta[0]))
            locked_context = corridor_local_door_context_allowed(
                bool(context.get("raw_is_corridor", False)),
                self.corridor_entry_forward_lock,
                locked_progress, locked_lateral_error,
                self.corridor_local_door_entry_lock_minimum_progress,
                self.corridor_centerline_membership_tolerance)
            if not locked_context:
                return False
            local_status = None
            with self.lock:
                local_status = dict(self.local_entry_status or {})
                # The detector publishes the raw and confirmed streams on
                # separate latched topics.  During startup the room-entry
                # stream can briefly lag behind the raw stream; do not lose a
                # valid side opening in that interval.  Both streams are
                # produced from the same local 8 m geometry and remain subject
                # to the checks below, so this is not a second planner.
                if not (local_status.get("candidates") or []):
                    local_status = dict(self.local_door_status or {})
                mission_start = self.mission_start_pose
            forward_displacement = 0.0
            if mission_start is not None and pose is not None:
                forward_displacement = math.hypot(
                    float(pose[0]) - float(mission_start[0]),
                    float(pose[1]) - float(mission_start[1]))
            local_door_evidence = any(
                bool(candidate.get("confirmed")) and
                bool(candidate.get("astar_reachable")) and
                bool(candidate.get("scan_lite_safe")) and
                0.75 <= float(candidate.get("width_m", 0.0)) <=
                self.local_door_maximum_semantic_width and
                self._local_door_has_room_area(candidate)
                for candidate in (local_status.get("candidates") or []))
            if not (forward_displacement >= 4.0 and
                    local_door_evidence):
                return False
            if (getattr(self, "corridor_guided_entry_door_context", False) and
                    getattr(self, "corridor_forward_axis_hint", None) is not None and
                    mission_start is not None):
                self.corridor_entry_forward_lock = True
                self._record_corridor_entry(
                    mission_start[:3], "upper_floor_guided_initial_pose",
                    elapsed_sec=0.0)
            self.corridor_established = True
            self.corridor_established_from_local_door = True
            self.corridor_established_pose = np.asarray(
                [float(pose[0]), float(pose[1]), float(pose[2])],
                dtype=float)
            self.corridor_confirmation_count = max(
                self.corridor_confirmation_count,
                self.corridor_confirmation_updates)
            self.corridor_direction_aligned = True
            # Seed the online station frame immediately.  Without this, the
            # old raw-corridor gate could establish the phase only after the
            # robot had already passed the first side openings.
            self.corridor_axis = np.asarray(
                context.get("axis", [1.0, 0.0]), dtype=float)
            self.corridor_axis /= max(1e-9, float(np.linalg.norm(
                self.corridor_axis)))
            centerline = context.get("centerline_point")
            self.corridor_anchor = np.asarray(
                centerline if centerline is not None else
                [float(pose[0]), float(pose[1])], dtype=float)
            self.corridor_station_axis = self.corridor_axis.copy()
            self.corridor_station_origin = self._corridor_station_entry_origin(
                self.corridor_anchor)
            self.corridor_sweep_history.append({
                "event": "CORRIDOR_ESTABLISHED_LOCAL_DOOR_EVIDENCE",
                "elapsed_sec": round(self.elapsed(), 3),
                "forward_displacement_m": round(forward_displacement, 3),
                "confirmation_updates": self.corridor_confirmation_count,
                "raw_is_corridor": bool(context.get("raw_is_corridor")),
                "entry_forward_lock_fallback": bool(
                    not context.get("raw_is_corridor", False)),
                "entry_lock_progress_m": round(locked_progress, 3),
                "entry_lock_lateral_error_m": round(
                    locked_lateral_error, 3),
                "local_door_evidence": True,
            })
            # Corridor-shaped geometry plus a side opening is sufficient to
            # establish the *axis*, but is not permission to immediately
            # select that opening.  In particular, at the lobby/corridor
            # transition the map can still contain a much cheaper path back
            # through the lobby.  Seed the station frame and require a small,
            # measured advance on this axis before side-door scheduling is
            # armed.
            self.corridor_door_arm_high_water = 0.0
            self.corridor_door_detection_armed = False
            self.corridor_door_forward_pending = True
            self.corridor_sweep_history.append({
                "event": "CORRIDOR_DOOR_SEARCH_FORWARD_COMMIT_PENDING",
                "elapsed_sec": round(self.elapsed(), 3),
                "relative_forward_progress_m": 0.0,
                "minimum_progress_m": self.corridor_door_forward_commit_distance,
                "policy_phase": "CORRIDOR_FORWARD_COMMIT_LOCAL_EVIDENCE",
            })
            return False
        progress = self._update_corridor_forward_high_water(pose, context)
        # With the zero global-progress setting, use a short relative forward
        # commit after the corridor confirmation.  Do not use a global y/x
        # segment or a layout-derived lobby boundary.
        if self.corridor_door_minimum_progress <= 0.0:
            relative_progress = (progress["high_water"]
                                 if progress is not None else 0.0)
            # Corridor confirmation is a phase boundary, not a doorway goal.
            # Capture the online forward station and require the normal
            # corridor planner to advance before side openings can take over.
            if self.corridor_door_arm_high_water is None:
                self.corridor_door_arm_high_water = relative_progress
                self.corridor_door_forward_pending = True
                return False
            if (self.corridor_established_pose is not None and
                    not self.corridor_established_from_local_door):
                axis = np.asarray(context.get("axis", [1.0, 0.0]),
                                  dtype=float)
                axis /= max(1e-9, float(np.linalg.norm(axis)))
                delta = np.asarray([float(pose[0]), float(pose[1])]) - \
                    np.asarray(self.corridor_established_pose[:2], dtype=float)
                if float(np.dot(delta, axis)) < self.corridor_door_post_latch_motion:
                    return False
            forward_delta = (relative_progress -
                             self.corridor_door_arm_high_water)
            if forward_delta < self.corridor_door_forward_commit_distance:
                return False
        elif (progress is None or
              progress["high_water"] < self.corridor_door_minimum_progress):
            return False
        else:
            relative_progress = progress["high_water"]
        self.corridor_door_detection_armed = True
        self.corridor_door_forward_pending = False
        self.corridor_sweep_history.append({
            "event": "CORRIDOR_DOOR_DETECTION_ARMED",
            "elapsed_sec": round(self.elapsed(), 3),
            "relative_forward_progress_m": relative_progress,
            "minimum_progress_m": self.corridor_door_minimum_progress,
            "policy_phase": "CORRIDOR_DOOR_SEARCH",
        })
        return True

    def _rearward_branch_is_paired_opposite(self, station, side,
                                            current_station):
        """Allow rearward work only for the unvisited opposite side nearby."""
        sign = (self.corridor_forward_station_sign
                if self.corridor_forward_station_sign is not None else 1.0)
        directed_delta = float(sign) * (float(station) -
                                        float(current_station))
        if directed_delta >= -self.corridor_new_branch_maximum_lookbehind:
            return True
        nearby = [
            branch for branch in self.branch_scheduler.branches
            if abs(float(branch.station) - float(station)) <=
            self.corridor_branch_station_tolerance]
        already_seen_same_side = any(
            int(branch.side) == int(side) and branch.state != "UNENTERED"
            for branch in nearby)
        explored_opposite = any(
            int(branch.side) == -int(side) and branch.state != "UNENTERED"
            for branch in nearby)
        return bool(explored_opposite and not already_seen_same_side)

    def _is_confirmed_paired_opposite_side(self, station, side):
        """Score, but never fabricate, the detected door opposite a visit."""
        nearby = [
            branch for branch in self.branch_scheduler.branches
            if abs(float(branch.station) - float(station)) <=
            self.corridor_branch_station_tolerance]
        return bool(
            any(int(branch.side) == -int(side) and
                branch.state != "UNENTERED" for branch in nearby) and
            not any(int(branch.side) == int(side) and
                    branch.state != "UNENTERED" for branch in nearby))

    def _on_execution_result(self, message):
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            return
        with self.lock:
            self.execution_results.append(payload)

    def _on_fastlio_registration_status(self, message):
        payload = self._decode_json_message(message)
        if payload is None:
            return
        healthy = bool(payload.get("healthy", True))
        invalid_count = int(payload.get("invalid_count", 0) or 0)
        with self.lock:
            self.fastlio_registration_healthy = healthy
            self.fastlio_registration_invalid_count = invalid_count
            self._fastlio_registration_status_sequence += 1
            self._localization_recovery_window.observe_status(
                self._fastlio_registration_status_sequence,
                healthy, invalid_count)
            self._sync_localization_recovery_diagnostics_locked()
            if not healthy and self.fastlio_registration_lost_elapsed is None:
                self.fastlio_registration_lost_elapsed = self.elapsed()

    def _on_stair_truth_return_gate(self, message):
        if bool(message.data):
            with self.lock:
                self.stair_truth_return_gate_reached = True

    def _on_locomotion_ready(self, message):
        ready = bool(message.data)
        with self.lock:
            was_ready = self.locomotion_ready
            self.locomotion_ready = ready
        # 下降沿: RL 状态被退出(FSM fall-safety → forcePassiveState)。
        # R28 实证: 撞墙打滑后 locomotion_ready 永久 false,探索死锁
        # (rescan 被拒 + 返程目标被拒 → GOAL_FAILURE)。自动恢复。
        if was_ready and not ready:
            self._start_locomotion_recovery()

    def _on_fixed_stand_ready(self, message):
        self._fixed_stand_ready = self._fixed_stand_ready or bool(message.data)

    def _publish_joy_button(self, index):
        message = Joy()
        message.header.stamp = rospy.Time.now()
        message.axes = [0.0] * 8
        message.buttons = [0] * 12
        message.buttons[index] = 1
        try:
            self._joy_pub.publish(message)
        except rospy.ROSException:
            pass

    def _start_locomotion_recovery(self):
        if self._locomotion_recovery_active:
            return
        # 恢复约需 25-45s(reset + stand + RL 接管),剩余时间太少则放弃。
        if self.maximum_duration - self.elapsed() < 90.0:
            rospy.logwarn("[LOCOMOTION] RL 被打断但剩余时间 <90s,放弃恢复")
            return
        self._locomotion_recovery_active = True
        threading.Thread(target=self._locomotion_recovery_worker,
                         daemon=True).start()

    def _locomotion_recovery_worker(self):
        try:
            rospy.logwarn("[LOCOMOTION] locomotion_ready=false,执行 FSM 恢复"
                          "(RESET→固定站→RL)")
            self._fixed_stand_ready = False
            # 1) RESET(按钮10): FSM 回出生位姿并进入 passive。
            # 高层(F2/F3)禁用 RESET(参数 locomotion_recovery_reset=false):
            # set_model_configuration 把机器人传回 F1 出生位姿,定位源瞬
            # 间跳跃 → localization_jump → FASTLIO_LOCALIZATION_LOST 终止。
            # 高层直接固定站→RL 重接,不传送。
            if self._locomotion_recovery_reset:
                for _ in range(3):
                    self._publish_joy_button(10)
                    time.sleep(0.5)
                # reset 内部 pause/unpause physics + set_model_configuration,
                # 需数秒完成
                time.sleep(4.0)
            # 2) 固定站(按钮1),等待 FixedStand 稳定
            self._publish_joy_button(1)
            stand_deadline = time.monotonic() + 15.0
            while not rospy.is_shutdown() and time.monotonic() < stand_deadline:
                if self._fixed_stand_ready:
                    break
                time.sleep(0.2)
            # 3) RL 接管(按钮3)重复,直到 locomotion_ready 重新锁存
            rl_deadline = time.monotonic() + 30.0
            while not rospy.is_shutdown() and time.monotonic() < rl_deadline:
                self._publish_joy_button(3)
                time.sleep(0.5)
                with self.lock:
                    ready = self.locomotion_ready
                if ready:
                    rospy.logwarn("[LOCOMOTION] 恢复成功: locomotion_ready=true,"
                                  "探索继续(地图保留,将从出生位姿重新规划)")
                    break
            else:
                rospy.logerr("[LOCOMOTION] 恢复超时: locomotion_ready 仍未恢复")
        except Exception as error:  # noqa: BLE001
            rospy.logerr("[LOCOMOTION] 恢复异常: %s", error)
        finally:
            self._locomotion_recovery_active = False

    def _on_map_saved(self, message):
        if message.data:
            with self.lock:
                self.map_saved_count += 1

    def _refresh_statistics(self):
        payload = self._load_json(self.statistics_file)
        if payload:
            with self.lock:
                self.map_statistics = payload
        return payload

    def _wait_slam(self):
        rospy.loginfo("[STARTUP] Waiting for FAST-LIO odometry and registered cloud")
        self._set_state("WAIT_FOR_SLAM")
        deadline = time.monotonic() + self.startup_timeout
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            with self.lock:
                ready = (self.pose is not None and self.first_cloud_elapsed is not None)
            if ready:
                rospy.loginfo("[STARTUP] FAST-LIO stream ready")
                self._write_all_logs()
                return True
            time.sleep(0.1)
        return False

    def _wait_map(self):
        rospy.loginfo("[STARTUP] Waiting for the first usable occupancy map")
        self._set_state("WAIT_FOR_MAP")
        deadline = time.monotonic() + self.startup_timeout
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            statistics = self._refresh_statistics() or {}
            with self.lock:
                ready = (self.grid is not None and
                         int(statistics.get("update_count", 0)) >= self.minimum_map_updates and
                         int(statistics.get("free_voxel_count", 0)) >= self.minimum_free_voxels and
                         os.path.isfile(self.map_file) and os.path.getsize(self.map_file) > 0 and
                         self.locomotion_ready)
            if ready:
                self.map_ready_elapsed = self.elapsed()
                rospy.loginfo("[STARTUP] Occupancy map ready")
                self._write_all_logs()
                return True
            time.sleep(0.2)
        return False

    def _write_planner_history_csv(self, path):
        with open(path, "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["# x", "y", "source"])
            for item in self.goal_history:
                if item.get("success"):
                    writer.writerow([item["goal"][0], item["goal"][1], "goal"])
                    frontier = item.get("frontier_center")
                    if isinstance(frontier, (list, tuple)) and len(frontier) >= 2:
                        writer.writerow([
                            float(frontier[0]), float(frontier[1]), "frontier"])
                elif (item.get("result") or {}).get("reason") != "start_footprint_blocked":
                    writer.writerow([item["goal"][0], item["goal"][1], "failed_goal"])
            # Goal-only history misses repeated arcs whose endpoints differ.
            # Subsample actual FAST-LIO motion and let FUEL treat it as visited
            # execution space while retaining its explicit transit fallback.
            with self.lock:
                trajectory = list(self.trajectory)
            last = None
            spacing = max(0.20, self.trajectory_history_spacing)
            for sample in trajectory:
                point = (float(sample[1]), float(sample[2]))
                if last is None or math.hypot(point[0] - last[0], point[1] - last[1]) >= spacing:
                    writer.writerow([point[0], point[1], "offline_execution"])
                    last = point

    def _corridor_context(self, pose=None, grid=None):
        """Estimate a local corridor axis from recent online odometry only."""
        with self.lock:
            pose = pose or self.pose
            grid = grid or self.grid
            samples = list(self.trajectory[-120:])
            grid_update = int(self.grid_update_count)
        if pose is None or grid is None:
            return None
        points = np.asarray([[item[1], item[2]] for item in samples], dtype=float)
        if self.corridor_axis is not None:
            # Once a reliable corridor is established, controller lateral
            # drift and room motion must not rotate the building axis.
            axis = np.asarray(self.corridor_axis, dtype=float)
            anisotropy = math.inf
        elif len(points) >= 8 and np.linalg.norm(points[-1] - points[0]) >= 1.0:
            centered = points - points.mean(axis=0)
            covariance = np.dot(centered.T, centered) / max(1, len(points) - 1)
            values, vectors = np.linalg.eigh(covariance)
            axis = vectors[:, int(np.argmax(values))]
            anisotropy = float(max(values) / max(1e-6, min(values)))
        else:
            axis = np.asarray([math.cos(pose[3]), math.sin(pose[3])], dtype=float)
            anisotropy = math.inf
        if self.corridor_axis is not None:
            # Once a corridor has been established, keep its axis across a
            # room visit.  The recent trajectory is then dominated by the
            # G3<->G4 breadth sweep; using that PCA axis after ROOM_EXIT makes
            # the corridor appear to run through the room and skips the
            # unvisited doorway on the opposite wall at the same station.
            # The stored axis came from the pre-entry corridor observations
            # and is exactly the reference CORRIDOR_RESUME must restore.
            axis = np.asarray(self.corridor_axis, dtype=float).copy()
        elif self.corridor_axis is None and len(points) >= 2:
            observed_motion = points[-1] - points[0]
            # Do not let centimetre-scale stationary FAST-LIO jitter choose
            # the sign of an otherwise symmetric corridor axis.  Until one
            # metre of real progress exists, the first valid FUEL goal below
            # remains responsible for direction alignment.
            if (float(np.linalg.norm(observed_motion)) >= 1.0 and
                    float(np.dot(axis, observed_motion)) < 0.0):
                axis = -axis
        if self.corridor_axis is None and self.mission_start_pose is not None:
            axis = np.asarray(orient_axis_away_from_origin(
                axis, (pose[0], pose[1]),
                (self.mission_start_pose[0], self.mission_start_pose[1])),
                dtype=float)
        axis = self._orient_corridor_axis_to_forward_hint(axis)
        heading = math.atan2(float(axis[1]), float(axis[0]))
        left_clearance = _ray_distance(
            grid, (pose[0], pose[1]), heading + 0.5 * math.pi, 4.0)
        right_clearance = _ray_distance(
            grid, (pose[0], pose[1]), heading - 0.5 * math.pi, 4.0)
        cross_width = left_clearance + right_clearance
        along_width = aperture_width(
            grid, (pose[0], pose[1]), heading + 0.5 * math.pi, 5.0)
        width_samples = []
        for offset in (-self.corridor_uniform_width_sample_offset,
                       0.0,
                       self.corridor_uniform_width_sample_offset):
            sample = (float(pose[0]) + offset * float(axis[0]),
                      float(pose[1]) + offset * float(axis[1]))
            cell = grid.world_to_cell(sample)
            if (cell is None or int(grid.data[cell[1], cell[0]]) != 0):
                continue
            sample_left = _ray_distance(
                grid, sample, heading + 0.5 * math.pi, 4.0)
            sample_right = _ray_distance(
                grid, sample, heading - 0.5 * math.pi, 4.0)
            width_samples.append(sample_left + sample_right)
        width_spread = ((max(width_samples) - min(width_samples))
                        if len(width_samples) >= 2 else math.inf)
        # Preserve the proven pre-entry corridor/door behaviour exactly.  The
        # three-station equal-width semantic is only a post-entry guard against
        # a furniture aisle inside a confirmed room being called a corridor.
        # Applying it while scanning the real corridor can suppress doorway
        # takeover when one sample happens to intersect an open doorway.
        scheduler = getattr(self, "room_scheduler", None)
        confirmed_inside_room = bool(
            scheduler is not None and
            scheduler.active_door is not None and
            scheduler.entry_confirmed)
        uniform_width = (
            not confirmed_inside_room or
            corridor_width_profile_is_uniform(
                width_samples, 4.2,
                self.corridor_maximum_width_spread, 3))
        raw_is_corridor = (
            0.85 <= cross_width <= 4.2 and
            along_width >= self.corridor_minimum_visible_length and
            along_width >= self.corridor_minimum_aspect_ratio * cross_width and
            anisotropy >= 1.5 and uniform_width)
        # Commit to the observed forward corridor as soon as it is genuinely
        # entered.  The three-update latch below is still required for door
        # activation; this earlier lock only prevents generic FUEL/recovery
        # from selecting the cheaper route back through the lobby meanwhile.
        if (not self.corridor_entry_forward_lock and raw_is_corridor and
                self.mission_start_pose is not None):
            displacement = math.hypot(
                float(pose[0]) - float(self.mission_start_pose[0]),
                float(pose[1]) - float(self.mission_start_pose[1]))
            if displacement >= self.corridor_entry_forward_lock_distance:
                relock = None
                if self.corridor_entry_axis_relock_parallel_wall_semantic:
                    relock = detect_parallel_wall_corridor(
                        grid, (float(pose[0]), float(pose[1])),
                        search_radius=
                            self.corridor_acquisition_semantic_search_radius,
                        minimum_width=0.85, maximum_width=4.2,
                        minimum_length=
                            self.corridor_acquisition_semantic_minimum_length,
                        angle_step_degrees=10.0)
                if relock is not None and bool(relock.get("robot_inside")):
                    relocked_axis = np.asarray(orient_axis_away_from_origin(
                        relock.get("axis", axis),
                        (float(pose[0]), float(pose[1])),
                        (float(self.mission_start_pose[0]),
                         float(self.mission_start_pose[1]))), dtype=float)
                    relocked_axis /= max(
                        1e-9, float(np.linalg.norm(relocked_axis)))
                    relocked_axis = self._orient_corridor_axis_to_forward_hint(
                        relocked_axis)
                    previous_axis = np.asarray(axis, dtype=float).copy()
                    observed_motion = (
                        float(pose[0]) - float(self.mission_start_pose[0]),
                        float(pose[1]) - float(self.mission_start_pose[1]))
                    consistency = corridor_axis_motion_consistency(
                        relocked_axis, observed_motion,
                        self.corridor_entry_axis_relock_maximum_motion_angle,
                        self.corridor_entry_axis_relock_minimum_motion)
                    common_relock_log = {
                        "elapsed_sec": round(self.elapsed(), 3),
                        "previous_axis": [float(previous_axis[0]),
                                          float(previous_axis[1])],
                        "candidate_axis": [float(relocked_axis[0]),
                                           float(relocked_axis[1])],
                        "observed_motion": [float(observed_motion[0]),
                                            float(observed_motion[1])],
                        "motion_distance_m": consistency.get(
                            "motion_distance_m"),
                        "motion_deviation_deg": (
                            math.degrees(consistency["deviation_rad"])
                            if consistency.get("deviation_rad") is not None
                            else None),
                        "maximum_motion_deviation_deg": math.degrees(
                            self.corridor_entry_axis_relock_maximum_motion_angle),
                        "width_m": float(relock.get("width", 0.0)),
                        "visible_length_m": float(
                            relock.get("visible_length", 0.0)),
                        "policy": "second_floor_instance_only",
                    }
                    if bool(consistency.get("consistent")):
                        relocked_axis = self._snap_axis_to_preferred_heading(
                            relocked_axis)
                        axis = relocked_axis
                        station_sync = (
                            self._synchronize_corridor_station_frame_after_relock(
                                relocked_axis,
                                relock.get("centerline_point",
                                           [float(pose[0]), float(pose[1])]),
                                pose))
                        common_relock_log.update({
                            "event":
                                "CORRIDOR_ENTRY_AXIS_PARALLEL_WALL_RELOCKED",
                            "axis": [float(axis[0]), float(axis[1])],
                            "consistency_reason": consistency.get("reason"),
                            "station_frame_synchronized": True,
                            "station_axis_before": station_sync["old_axis"],
                            "station_axis_after": station_sync["new_axis"],
                            "station_origin_before": station_sync["old_origin"],
                            "station_origin_after": station_sync["new_origin"],
                        })
                    else:
                        # Keep the trajectory/PCA axis that already agrees
                        # with actual forward travel.  In run64 this rejects
                        # the spurious +20 deg wall raster and retains -10 deg,
                        # preventing forward corridor space from becoming a
                        # false side-room portal.
                        self.corridor_axis = previous_axis.copy()
                        common_relock_log.update({
                            "event":
                                "CORRIDOR_ENTRY_AXIS_PARALLEL_WALL_RELOCK_REJECTED",
                            "axis": [float(previous_axis[0]),
                                     float(previous_axis[1])],
                            "consistency_reason": consistency.get("reason"),
                            "action": "retain_online_motion_axis",
                        })
                    self.corridor_sweep_history.append(common_relock_log)
                self.corridor_entry_forward_lock = True
                self.corridor_entry_anchor = (
                    float(pose[0]), float(pose[1]), float(pose[2]))
                self._record_corridor_entry(
                    (float(pose[0]), float(pose[1]), float(pose[2])),
                    "CORRIDOR_ENTRY_FORWARD_LOCKED",
                    elapsed_sec=round(self.elapsed(), 3),
                    append_history=False)
                self.corridor_sweep_history.append({
                    "event": "CORRIDOR_ENTRY_FORWARD_LOCKED",
                    "elapsed_sec": round(self.elapsed(), 3),
                    "displacement_m": round(displacement, 3),
                    "axis": [float(axis[0]), float(axis[1])],
                    "anchor": list(self.corridor_entry_anchor[:2]),
                    "reason": "suppress_lobby_backtrack_until_terminal_or_return",
                })
        # Evaluate local doorway evidence before the raw wall-width latch.
        # A real doorway can make one side ray fail the corridor test, so
        # waiting for raw_is_corridor here causes the manager to pass the door
        # and only arm the room scheduler much later.
        if not self.corridor_established:
            self._update_corridor_door_phase(
                pose, {"axis": axis, "raw_is_corridor": raw_is_corridor,
                       "centerline_point": [float(pose[0]), float(pose[1])]})
        if (grid_update != self.corridor_last_confirmation_grid_update and
                not self.corridor_established):
            self.corridor_last_confirmation_grid_update = grid_update
            self.corridor_confirmation_count = (
                self.corridor_confirmation_count + 1
                if raw_is_corridor else 0)
            if (self.corridor_confirmation_count >=
                    self.corridor_confirmation_updates):
                self.corridor_established = True
                self.corridor_established_pose = np.asarray(
                    [float(pose[0]), float(pose[1]), float(pose[2])],
                    dtype=float)
                established_motion = (
                    float(np.linalg.norm(points[-1] - points[0]))
                    if len(points) >= 2 else 0.0)
                self.corridor_direction_aligned = established_motion >= 1.0
                self.corridor_sweep_history.append({
                    "event": "CORRIDOR_ESTABLISHED",
                    "elapsed_sec": round(self.elapsed(), 3),
                    "confirmation_updates": self.corridor_confirmation_count,
                    "axis": [float(axis[0]), float(axis[1])],
                    "cross_width_m": cross_width,
                    "visible_length_m": along_width,
                    "axis_source": (
                        "mission_start_to_current_odometry" if
                        self.corridor_direction_aligned else
                        "unoriented_geometry_pending_first_goal"),
                })
        # Once a confirmed side opening has established the local corridor
        # phase, retain corridor semantics even if the opening itself causes
        # one wall-width sample to fail the raw corridor test.  This prevents
        # a real door from being lost while the robot crosses its aperture.
        is_corridor = bool(
            self.corridor_established and
            (raw_is_corridor or self.corridor_established_from_local_door))
        if is_corridor:
            if self.corridor_axis is None:
                self.corridor_axis = np.asarray(axis, dtype=float)
            allow_anchor_update = True
            if self.corridor_anchor is not None:
                current = np.asarray(
                    [float(pose[0]), float(pose[1])], dtype=float)
                old_center = self.corridor_anchor + np.dot(
                    current - self.corridor_anchor, axis) * axis
                allow_anchor_update = (
                    float(np.linalg.norm(current - old_center)) <=
                    self.corridor_centerline_membership_tolerance)
            # Both nearby walls define the corridor centre.  At a doorway one
            # side expands beyond this bound, so retain the previous centre
            # instead of being pulled into the room opening.
            bilateral_center = None
            if (allow_anchor_update and
                    max(left_clearance, right_clearance) <= 2.30):
                bilateral_center = np.asarray(corridor_wall_midpoint(
                    pose, axis, left_clearance, right_clearance), dtype=float)
                self.corridor_anchor = bilateral_center
            elif self.corridor_anchor is None:
                self.corridor_anchor = np.asarray(
                    [float(pose[0]), float(pose[1])], dtype=float)
            if (self.corridor_station_axis is None and
                    self.corridor_anchor is not None):
                # Keep branch station and side labels independent of rolling
                # centreline updates and a later return-sweep axis reversal.
                self.corridor_station_axis = np.asarray(axis, dtype=float).copy()
                self.corridor_station_axis /= max(
                    1e-9, float(np.linalg.norm(self.corridor_station_axis)))
                self.corridor_station_origin = self._corridor_station_entry_origin(
                    self.corridor_anchor)
                self.corridor_station_centerline_calibrated = bool(
                    bilateral_center is not None)
            elif (bilateral_center is not None and
                  not self.corridor_station_centerline_calibrated and
                  self.room_scheduler.active_door is None and
                  not any(bool(door.visited) for door in
                          self.room_scheduler.detector.doors)):
                old_origin = np.asarray(
                    self.corridor_station_origin, dtype=float)
                new_origin = np.asarray(rebase_corridor_line_laterally(
                    old_origin, axis, bilateral_center), dtype=float)
                correction = float(np.linalg.norm(new_origin - old_origin))
                self.corridor_station_origin = new_origin
                self.corridor_station_centerline_calibrated = True
                self.corridor_sweep_history.append({
                    "event": "CORRIDOR_STATION_CENTERLINE_BILATERAL_CALIBRATED",
                    "elapsed_sec": round(self.elapsed(), 3),
                    "lateral_correction_m": round(correction, 3),
                    "old_origin": old_origin.tolist(),
                    "new_origin": new_origin.tolist(),
                    "left_clearance_m": round(float(left_clearance), 3),
                    "right_clearance_m": round(float(right_clearance), 3),
                })
        centerline_point = np.asarray([float(pose[0]), float(pose[1])])
        if self.corridor_anchor is not None:
            centerline_point = self.corridor_anchor + np.dot(
                centerline_point - self.corridor_anchor, axis) * axis
        result = {
            "is_corridor": is_corridor, "axis": axis,
            "raw_is_corridor": raw_is_corridor,
            "corridor_established": self.corridor_established,
            "confirmation_count": self.corridor_confirmation_count,
            "heading": heading, "cross_width": cross_width,
            "along_width": along_width, "anisotropy": anisotropy,
            "width_samples": width_samples,
            "width_spread": width_spread,
            "uniform_width": uniform_width,
            "left_clearance": left_clearance,
            "right_clearance": right_clearance,
            "centerline_point": centerline_point,
        }
        # Let a raw corridor-shaped update plus a strongly confirmed local
        # side opening establish the corridor phase.  Calling this only after
        # ``is_corridor`` became true created a circular gate: ``is_corridor``
        # requires ``corridor_established``, while the local-door fallback in
        # _update_corridor_door_phase() is precisely what establishes it when
        # the three-frame raw counter flickers at open doorways.  Run28 saw
        # both physical room portals at x~=14.6, but discarded them all and
        # retraced to the lobby because this fallback was unreachable.
        # R29 F3: a large corridor replacement waypoint can put the robot on
        # the corridor edge / door-zone while raw_is_corridor briefly fails,
        # so the arming call below was gated off forever and door detection
        # never armed (0 rooms -> TIME_LIMIT).  Once the corridor is
        # established and not yet armed, always re-run the arming check; it
        # is guarded internally by centerline/high-water/post-latch tests and
        # returns None off-corridor, so it cannot arm while inside a room.
        if (raw_is_corridor or is_corridor or
                (self.corridor_established and
                 not self.corridor_door_detection_armed)):
            self._update_corridor_door_phase(pose, result)
        return result

    def _snap_axis_to_preferred_heading(self, axis):
        """Return axis aligned to the guide heading when it is close enough.

        The parallel-wall relock detector searches wall orientation on a
        10-deg raster, so a geometrically pure N/S corridor is often reported
        as +-10 deg off axis.  A few degrees of tilt is harmless locally, but
        multiplied by a 6-7 m corridor advance it pushes the sweep target
        past the corridor wall; the only A* route then leads through a room
        doorway (run R11) and the robot wedges in the door.  When a preferred
        guide heading exists and the relock axis is within tolerance, snap
        the axis to the guide heading - the floors are axis-aligned by
        construction, so this removes raster noise without using any layout
        coordinates.
        """
        pref = self.corridor_acquisition_preferred_heading
        if pref is None or not math.isfinite(float(pref)):
            return axis
        pref_vec = np.asarray(
            [math.cos(pref), math.sin(pref)], dtype=float)
        vec = np.asarray(axis, dtype=float)
        if float(np.dot(vec, pref_vec)) < 0.0:
            vec = -vec
        dot = float(np.dot(vec, pref_vec))
        tolerance = float(self.corridor_axis_preferred_snap_tolerance_deg)
        if dot >= math.cos(math.radians(tolerance)):
            return pref_vec.copy()
        return np.asarray(axis, dtype=float)

    def _corridor_forward_plan(self, cycle, pose, grid, axis,
                               maximum_advance=None):
        axis = np.asarray(axis, dtype=float)
        # Use the immutable station-frame origin for corridor transit.  The
        # rolling wall midpoint is useful for local perception, but a wide
        # doorway can pull it laterally into a room.  During run25 retrace that
        # produced a replacement at (13.44,-0.93), physically re-entering
        # room 1.  The station origin and axis were latched from the corridor
        # before any room visit and therefore define the stable transit line.
        anchor = (np.asarray(self.corridor_station_origin, dtype=float)
                  if self.corridor_station_origin is not None else
                  np.asarray(self.corridor_anchor, dtype=float)
                  if self.corridor_anchor is not None else
                  np.asarray([pose[0], pose[1]], dtype=float))
        current = np.asarray([pose[0], pose[1]], dtype=float)
        centerline = anchor + np.dot(current - anchor, axis) * axis
        # A corridor established from a fresh local doorway needs only a
        # short measured forward motion before doorway takeover is safe.  Do
        # not send the usual 7 m sweep first: its confirming candidates age
        # out long before that goal completes, which skips every near door.
        forward_commit_pending = bool(
            self.corridor_door_forward_pending and
            (self.corridor_established_from_local_door or
             self.corridor_short_door_commit_after_establishment) and
            not self.corridor_door_detection_armed)
        if forward_commit_pending:
            distances = [max(0.55,
                             self.corridor_door_forward_commit_distance)]
            lateral_offsets = [0.0]
        else:
            requested_forward_distance = self.corridor_forward_distance
            exited_rooms = sum(
                bool(door.visited)
                for door in self.room_scheduler.detector.doors)
            far_pair_search = bool(
                getattr(self, "corridor_far_pair_search_advance", 0.0) >
                0.05 and
                exited_rooms == getattr(
                    self, "corridor_far_pair_search_after_exits", 2) and
                self.room_scheduler.active_door is None and
                not self._missing_room_retrace_active() and
                not self.corridor_terminal_return_latched)
            if far_pair_search:
                unclamped_distance = requested_forward_distance
                requested_forward_distance = min(
                    requested_forward_distance,
                    getattr(self, "corridor_far_pair_search_advance",
                            requested_forward_distance))
                self.corridor_sweep_history.append({
                    "event": "CORRIDOR_FAR_PAIR_SEARCH_ADVANCE_CLAMPED",
                    "elapsed_sec": round(self.elapsed(), 3),
                    "exited_rooms": exited_rooms,
                    "unclamped_advance_m": float(unclamped_distance),
                    "maximum_advance_m": float(requested_forward_distance),
                })
            if maximum_advance is not None:
                requested_forward_distance = min(
                    requested_forward_distance,
                    max(0.0, float(maximum_advance)))
            if requested_forward_distance <= 0.05:
                return None, None
            minimum_advance = min(
                self.corridor_forward_recovery_minimum_advance,
                requested_forward_distance)
            distances = list(np.arange(
                requested_forward_distance,
                minimum_advance - 0.01,
                -0.5))
            if (not distances or
                distances[-1] >
                    minimum_advance + 0.05):
                distances.append(minimum_advance)
            lateral_offsets = [0.0]
            if self.corridor_forward_lateral_search > 0.0:
                half = 0.5 * self.corridor_forward_lateral_search
                lateral_offsets.extend((half, -half,
                                        self.corridor_forward_lateral_search,
                                        -self.corridor_forward_lateral_search))
        normal = np.asarray([-axis[1], axis[0]], dtype=float)
        for distance in distances:
            for lateral_offset in lateral_offsets:
                target = (centerline + float(distance) * axis +
                          float(lateral_offset) * normal)
                point = (float(target[0]), float(target[1]))
                result = astar_safe_path(
                    grid, (pose[0], pose[1]), point, self.clearance,
                    self.reached_tolerance, allow_blocked_start=True)
                if result.get("success"):
                    goal = {
                        "position": [point[0], point[1], pose[2]],
                        "information_gain": 0.0, "score": 0.0,
                        "source": ("corridor_door_forward_commit"
                                   if forward_commit_pending else
                                   "corridor_sweep"),
                        "scheduler_phase": "CORRIDOR_SWEEP",
                        "corridor_advance_m": float(distance),
                        "corridor_lateral_offset_m":
                            float(lateral_offset),
                        "corridor_incremental_recovery": bool(
                            float(distance) < self.corridor_minimum_advance),
                        "corridor_reversed": self.corridor_reversed,
                        "_preplanned_path_result": result,
                    }
                    return goal, result
        return None, None

    def _plan_corridor_acquisition_goal(self, cycle, pose, grid, context):
        """Move straight out of the lobby until corridor semantics latch.

        This intentionally uses no layout/lobby coordinates.  The online
        heading/corridor axis supplies the forward ray, and accepting only a
        near-straight A* path prevents an apparently forward target from
        routing around the lobby and becoming another generic exploration
        loop.
        """
        if (not self.enable_corridor_acquisition or
                self.corridor_established or pose is None or grid is None or
                context is None):
            return None
        axis = np.asarray(context.get("axis", []), dtype=float)
        if axis.shape != (2,) or not np.all(np.isfinite(axis)):
            return None
        axis /= max(1e-9, float(np.linalg.norm(axis)))
        start = (float(pose[0]), float(pose[1]))
        semantic = None
        semantic_axis = None
        if self.corridor_acquisition_parallel_wall_semantic:
            semantic = detect_parallel_wall_corridor(
                grid, start,
                search_radius=self.corridor_acquisition_semantic_search_radius,
                minimum_width=0.85, maximum_width=4.2,
                minimum_length=
                    self.corridor_acquisition_semantic_minimum_length,
                angle_step_degrees=10.0)
            if semantic is not None:
                semantic_axis = np.asarray(
                    semantic.get("direction", []), dtype=float)
                if (semantic_axis.shape != (2,) or
                        not np.all(np.isfinite(semantic_axis)) or
                        float(np.linalg.norm(semantic_axis)) < 1e-6):
                    semantic_axis = None
                else:
                    semantic_axis /= float(np.linalg.norm(semantic_axis))
        preferred_axis = None
        if self.corridor_acquisition_preferred_heading is not None:
            preferred_heading = self.corridor_acquisition_preferred_heading
            if math.isfinite(preferred_heading):
                preferred_axis = np.asarray([
                    math.cos(preferred_heading),
                    math.sin(preferred_heading)], dtype=float)
        axes = (([semantic_axis] if semantic_axis is not None else []) +
                ([preferred_axis] if preferred_axis is not None else []) +
                [axis])
        if self.corridor_acquisition_heading_search:
            sample_count = max(8, self.corridor_acquisition_heading_samples)
            radial_axes = [np.asarray([
                math.cos(2.0 * math.pi * index / sample_count),
                math.sin(2.0 * math.pi * index / sample_count)], dtype=float)
                    for index in range(sample_count)]
            radial_axes.sort(key=lambda candidate: _ray_distance(
                grid, start, math.atan2(candidate[1], candidate[0]),
                self.corridor_acquisition_heading_range), reverse=True)
            axes.extend(radial_axes)
        # Preserve ordering while removing near-identical headings.  The
        # semantic direction receives first refusal, then the ordinary local
        # axis and longest observed-free radial rays remain safe fallbacks.
        unique_axes = []
        for candidate in axes:
            if candidate is None:
                continue
            candidate = np.asarray(candidate, dtype=float)
            candidate /= max(1e-9, float(np.linalg.norm(candidate)))
            candidate = self._orient_corridor_axis_to_forward_hint(candidate)
            if any(float(np.dot(candidate, old)) > 0.985
                   for old in unique_axes):
                continue
            unique_axes.append(candidate)
        axes = unique_axes
        # While a guide heading is preferred, a candidate pointing back down
        # the arrival path (R8k: south toward the already-mapped stair core)
        # must not win by default when the true corridor forward (north,
        # still unmapped) has no A* path yet.  Rejecting near-opposite axes
        # keeps acquisition from locking the arrival direction; the fallback
        # is then the generic frontier planner, which in R8j picked north and
        # mapped the corridor before the axis relock.
        if preferred_axis is not None:
            axes = [candidate for candidate in axes
                    if float(np.dot(candidate, preferred_axis)) > -0.7]
        selected = None
        for candidate_axis in axes:
            for distance in np.arange(
                    self.corridor_acquisition_distance,
                    self.corridor_acquisition_minimum_advance - 0.01, -0.5):
                target = (start[0] + float(distance) * float(candidate_axis[0]),
                          start[1] + float(distance) * float(candidate_axis[1]))
                result = astar_safe_path(
                    grid, start, target, self.clearance, self.reached_tolerance,
                    allow_blocked_start=True)
                if not result.get("success"):
                    continue
                path = list(result.get("path") or [])
                path_length = sum(
                    math.hypot(float(b[0]) - float(a[0]),
                               float(b[1]) - float(a[1]))
                    for a, b in zip(path, path[1:]))
                if path_length > (float(distance) *
                                  self.corridor_acquisition_maximum_detour_ratio):
                    continue
                selected = (candidate_axis, distance, target, result,
                            path_length)
                break
            if selected is not None:
                break
        if selected is not None:
            candidate_axis, distance, target, result, path_length = selected
            goal = {
                "position": [target[0], target[1], float(pose[2])],
                "information_gain": 0.0, "score": 0.0,
                "source": "corridor_acquisition",
                "scheduler_phase": "SEEK_CORRIDOR_FORWARD",
                "corridor_advance_m": float(distance),
                "corridor_axis": [float(candidate_axis[0]),
                                  float(candidate_axis[1])],
                "_preplanned_path_result": result,
            }
            self.corridor_sweep_history.append({
                "event": "CORRIDOR_ACQUISITION_FORWARD_SELECTED",
                "elapsed_sec": round(self.elapsed(), 3),
                "goal": list(goal["position"]),
                "advance_m": round(float(distance), 3),
                "path_length_m": round(path_length, 3),
                "heading_search": bool(
                    self.corridor_acquisition_heading_search),
                "preferred_heading_used": bool(
                    preferred_axis is not None and
                    float(np.dot(candidate_axis, preferred_axis)) > 0.985),
                "parallel_wall_semantic": bool(
                    semantic is not None and
                    np.dot(candidate_axis, semantic_axis) > 0.985
                    if semantic_axis is not None else False),
                "semantic_score": (round(float(semantic["score"]), 3)
                                   if semantic is not None else None),
                "semantic_width_m": (round(float(semantic["width"]), 3)
                                     if semantic is not None else None),
                "semantic_visible_length_m": (
                    round(float(semantic["visible_length"]), 3)
                    if semantic is not None else None),
            })
            return goal
        self.corridor_sweep_history.append({
            "event": "CORRIDOR_ACQUISITION_FORWARD_UNAVAILABLE",
            "elapsed_sec": round(self.elapsed(), 3),
            "reason": "no_direct_online_forward_path",
        })
        return None

    def _on_established_corridor_centerline(self, pose, corridor) -> bool:
        """Keep an open doorway from masquerading as physical room entry."""
        if not corridor:
            return False
        if self.corridor_axis is not None and self.corridor_anchor is not None:
            # Once the building corridor has been established, it is the
            # global reference.  Narrow aisles between room furniture may
            # satisfy the local corridor-width test, but must not overwrite
            # physical room-entry state or trigger corridor side scheduling.
            center = np.asarray(corridor["centerline_point"], dtype=float)
            current = np.asarray([float(pose[0]), float(pose[1])], dtype=float)
            return float(np.linalg.norm(current - center)) <= \
                self.corridor_centerline_membership_tolerance
        return bool(corridor.get("is_corridor"))

    def _on_station_corridor_centerline(self, pose, tolerance=None) -> bool:
        """Require transit decisions to remain near the immutable station line."""
        # Before the first verified room exit the station line is still the
        # original oblique acquisition estimate.  Retain the established
        # first-room behaviour until stronger exit evidence has rebased it.
        if not self.corridor_station_centerline_rebased:
            return True
        if (pose is None or self.corridor_station_origin is None or
                self.corridor_station_axis is None):
            return False
        limit = (self.corridor_centerline_membership_tolerance
                 if tolerance is None else float(tolerance))
        return corridor_line_lateral_error(
            pose, self.corridor_station_origin,
            self.corridor_station_axis) <= limit

    def _remember_visited_region_sample(self, point, side, branch_id):
        """Persist online map-frame evidence of physical side-region visits."""
        sample = (float(point[0]), float(point[1]))
        stable_side = 1 if int(side) >= 0 else -1
        for item in reversed(self.visited_region_samples[-300:]):
            if (item["side"] == stable_side and
                    math.hypot(item["point"][0] - sample[0],
                               item["point"][1] - sample[1]) <
                    self.visited_region_sample_spacing):
                return
        self.visited_region_samples.append({
            "point": sample,
            "side": stable_side,
            "branch_id": branch_id,
            "elapsed_sec": round(self.elapsed(), 3),
        })

    def _visited_side_component(self, target, side, grid):
        axis = (self.corridor_station_axis
                if self.corridor_station_axis is not None else
                self.corridor_axis)
        origin = (self.corridor_station_origin
                  if self.corridor_station_origin is not None else
                  self.corridor_anchor)
        if axis is None or origin is None or grid is None:
            return {
                "matched": False, "reason": "corridor_frame_unavailable",
                "matched_branch_ids": [],
            }
        stable_side = 1 if int(side) >= 0 else -1
        direction = np.asarray(axis, dtype=float)
        direction /= max(1e-9, float(np.linalg.norm(direction)))
        frame_origin = np.asarray(origin[:2], dtype=float)
        target_station = float(np.dot(
            np.asarray(target[:2], dtype=float) - frame_origin, direction))
        references = [
            item for item in self.visited_region_samples
            if (item["side"] == stable_side and
                abs(float(np.dot(
                    np.asarray(item["point"][:2], dtype=float) -
                    frame_origin, direction)) - target_station) <=
                self.visited_region_maximum_station_separation)]
        if not references:
            return {
                "matched": False,
                "reason": "no_nearby_visited_station",
                "matched_branch_ids": [],
                "target_station": target_station,
            }
        result = side_region_component_match(
            grid, (float(target[0]), float(target[1])),
            [item["point"] for item in references],
            axis, (float(origin[0]), float(origin[1])), stable_side,
            self.visited_region_component_minimum_lateral)
        matched_ids = []
        for index in result.get("matched_indices", []):
            if 0 <= index < len(references):
                old_id = references[index].get("branch_id")
                branch = self.branch_scheduler.find(old_id)
                branch_id = branch.branch_id if branch is not None else old_id
                if branch_id and branch_id not in matched_ids:
                    matched_ids.append(branch_id)
        result["matched_branch_ids"] = matched_ids
        result["target_station"] = target_station
        result["nearby_reference_count"] = len(references)
        return result

    def _visited_component_for_corridor_goal(self, target, pose, grid):
        """Check a global target only while the robot is on the corridor.

        Samples collected during the current room visit are intentionally not
        used to suppress that room's own coverage goals.  The guard at the
        corridor is what prevents a later global frontier from routing back
        into an already entered side component.
        """
        context = self._corridor_context(pose, grid)
        if not self._on_established_corridor_centerline(pose, context):
            return {
                "matched": False, "reason": "robot_not_in_corridor",
                "matched_branch_ids": [],
            }
        axis = (self.corridor_station_axis
                if self.corridor_station_axis is not None else
                self.corridor_axis)
        origin = (self.corridor_station_origin
                  if self.corridor_station_origin is not None else
                  self.corridor_anchor)
        if axis is None or origin is None:
            return {
                "matched": False, "reason": "corridor_frame_unavailable",
                "matched_branch_ids": [],
            }
        direction = np.asarray(axis, dtype=float)
        direction /= max(1e-9, float(np.linalg.norm(direction)))
        normal = np.asarray([-direction[1], direction[0]], dtype=float)
        lateral = float(np.dot(
            np.asarray(target[:2], dtype=float) -
            np.asarray(origin[:2], dtype=float), normal))
        if abs(lateral) < self.visited_region_component_minimum_lateral:
            return {
                "matched": False, "reason": "goal_inside_corridor_strip",
                "matched_branch_ids": [],
            }
        return self._visited_side_component(
            target, 1 if lateral >= 0.0 else -1, grid)

    def _corridor_side_coverage_goal(self, candidates, scores, pose, grid,
                                     corridor, raw_frontiers=None):
        """Prefer a reachable side-area entry over a distant corridor end.

        This uses only online occupancy frontiers and the locally estimated
        corridor axis.  A frontier beyond either corridor wall is coverage
        debt.  Its longitudinal projection identifies an opening station; an
        A* verified known-free point beyond that wall commits the robot into
        the wider area.  No doorway detector, room polygon, or layout truth is
        used.
        """
        if (not self.corridor_side_coverage_priority or not corridor or
                not self.corridor_door_detection_armed or
                not self._on_established_corridor_centerline(pose, corridor) or
                grid is None):
            return None
        axis, normal = self._stable_corridor_axes(corridor["axis"])
        current = np.asarray([float(pose[0]), float(pose[1])], dtype=float)
        # Do not gate side-opening registration by accumulated global-axis
        # progress. FAST-LIO scale drift along a long corridor makes that
        # quantity unreliable; local bilateral geometry is sufficient.
        center = np.asarray(corridor["centerline_point"], dtype=float)
        score_by_id = {
            int(item.get("candidate_id", -1)): item
            for item in (scores.get("scores") or [])
        }
        # The FUEL connectivity funnel can reject a room frontier before its
        # own viewpoint is connected, even though a nearer known-free doorway
        # staging point is A*-reachable.  Branch *discovery* must therefore
        # also inspect every raw frontier cluster.  Execution remains strict:
        # the derived endpoint below must be observed free and pass A* (then
        # SCAN-lite in the normal execution chain).
        combined_candidates = list(candidates or [])
        represented_frontiers = {
            int(item.get("frontier_id", -1)) for item in combined_candidates}
        synthetic_id = -1000000
        for frontier in raw_frontiers or []:
            frontier_id = int(frontier.get("id", -1))
            center_point = frontier.get("center") or []
            if (frontier_id in represented_frontiers or
                    len(center_point) < 2 or
                    int(frontier.get("size", 0)) <
                    self.corridor_side_raw_frontier_minimum_size):
                continue
            combined_candidates.append({
                "id": synthetic_id,
                "frontier_id": frontier_id,
                "frontier_center": center_point,
                "frontier_size": int(frontier.get("size", 0)),
                "reachable": True,
                "execution_duplicate": False,
                "raw_frontier_evidence": True,
            })
            synthetic_id -= 1
        ranked = sorted(
            combined_candidates,
            key=lambda item: float(
                score_by_id.get(int(item.get("id", -1)), {}).get(
                    "score", -math.inf)),
            reverse=True)
        qualified = []
        rejected_visited_components = set()
        station_axis = (
            np.asarray(self.corridor_station_axis, dtype=float)
            if self.corridor_station_axis is not None else axis)
        station_origin = (
            np.asarray(self.corridor_station_origin, dtype=float)
            if self.corridor_station_origin is not None else center)
        station_normal = np.asarray(
            [-station_axis[1], station_axis[0]], dtype=float)
        for candidate in ranked:
            if (not candidate.get("reachable", False) or
                    candidate.get("execution_duplicate", False)):
                continue
            frontier = candidate.get("frontier_center") or []
            if len(frontier) < 2:
                continue
            frontier_point = np.asarray(
                [float(frontier[0]), float(frontier[1])], dtype=float)
            longitudinal = float(np.dot(frontier_point - current, axis))
            station = center + longitudinal * axis
            signed_lateral = float(np.dot(frontier_point - station, normal))
            if not corridor_side_frontier_eligible(
                    longitudinal, signed_lateral,
                    self.corridor_side_frontier_lateral,
                    self.corridor_side_maximum_longitudinal):
                continue
            side = 1.0 if signed_lateral >= 0.0 else -1.0
            station_coordinate = float(np.dot(
                station - station_origin, station_axis))
            stable_side = (
                1 if float(np.dot(frontier_point - station,
                                  station_normal)) >= 0.0 else -1)
            # Discovery and execution are deliberately separated.  A room-side
            # frontier can initially expose only a thin known-free strip.  If
            # branch registration required a 1.45--2.10 m endpoint immediately,
            # that side would never enter persistent memory and could never be
            # approached to grow the map.  Probe deep targets first, then a
            # known-free staging target just beyond the observed corridor wall.
            corridor_half_width = 0.5 * float(
                corridor.get("cross_width", 0.0) or 0.0)
            staging_depth = max(
                self.corridor_side_staging_minimum_depth,
                corridor_half_width + max(0.10, grid.resolution),
                self.corridor_centerline_membership_tolerance + 0.15)
            depths = []
            for value in (self.corridor_side_entry_depth,
                          max(1.65, self.corridor_side_entry_depth - 0.30),
                          1.45, staging_depth):
                value = float(value)
                if all(abs(value - old) > 1e-6 for old in depths):
                    depths.append(value)
            offset_count = int(math.floor(
                self.corridor_side_station_search_radius /
                self.corridor_side_station_search_step + 1e-9))
            station_offsets = [0.0]
            for index in range(1, offset_count + 1):
                delta = index * self.corridor_side_station_search_step
                station_offsets.extend((delta, -delta))
            selected_target = None
            selected_depth = None
            selected_station = None
            selected_path = None
            for depth in depths:
                for station_offset in station_offsets:
                    target_station = station + station_offset * axis
                    target = target_station + side * depth * normal
                    cell = grid.world_to_cell(
                        (float(target[0]), float(target[1])))
                    if (cell is None or
                            int(grid.data[cell[1], cell[0]]) != 0):
                        continue
                    path = astar_safe_path(
                        grid, (float(pose[0]), float(pose[1])),
                        (float(target[0]), float(target[1])),
                        self.clearance, self.reached_tolerance)
                    if not path.get("success"):
                        continue
                    selected_target = target
                    selected_depth = depth
                    selected_station = target_station
                    selected_path = list(path.get("path") or [])
                    break
                if selected_target is not None:
                    break
            if selected_target is not None:
                target = selected_target
                depth = float(selected_depth)
                station = selected_station
                station_coordinate = float(np.dot(
                    station - station_origin, station_axis))
                # The frontier's longitudinal projection is not necessarily
                # its doorway. If A* reaches that frontier by leaving the
                # corridor at another station, remember the actual crossing
                # and a local just-inside endpoint. Otherwise a persisted
                # opposite-side branch can later command a wall crossing.
                portal = corridor_portal_from_reachable_path(
                    selected_path, station_origin, station_axis, stable_side,
                    minimum_inside_lateral=max(
                        self.semantic_side_door_minimum_penetration,
                        min(depth, 1.45)))
                if (portal is not None and
                        abs(float(portal["station"]) -
                            station_coordinate) >
                        max(0.90,
                            0.5 * self.corridor_branch_station_tolerance)):
                    original_station = station_coordinate
                    original_target = (float(target[0]), float(target[1]))
                    station_coordinate = float(portal["station"])
                    station = np.asarray(portal["centerline"], dtype=float)
                    target = np.asarray(portal["inside_point"], dtype=float)
                    depth = float(portal["inside_lateral"])
                    self.corridor_sweep_history.append({
                        "event": "SIDE_FRONTIER_PORTAL_STATION_NORMALIZED",
                        "elapsed_sec": round(self.elapsed(), 3),
                        "frontier_center": [float(frontier_point[0]),
                                            float(frontier_point[1])],
                        "original_station_m": original_station,
                        "portal_station_m": station_coordinate,
                        "station_shift_m": abs(
                            station_coordinate - original_station),
                        "original_target": list(original_target),
                        "local_inside_target": [float(target[0]),
                                                float(target[1])],
                        "path_index": int(portal["path_index"]),
                    })
                current_station_coordinate = self._corridor_station(
                    (pose[0], pose[1]))
                if not self._rearward_branch_is_paired_opposite(
                        station_coordinate, stable_side,
                        current_station_coordinate):
                    self.corridor_sweep_history.append({
                        "event": "REARWARD_NEW_SIDE_BRANCH_REJECTED",
                        "elapsed_sec": round(self.elapsed(), 3),
                        "candidate_station_m": station_coordinate,
                        "current_station_m": current_station_coordinate,
                        "side": stable_side,
                        "frontier_center": [float(frontier_point[0]),
                                            float(frontier_point[1])],
                    })
                    continue
                component = self._visited_side_component(
                    target, stable_side, grid)
                if component.get("matched"):
                    key = tuple(component.get("matched_branch_ids", []))
                    if key not in rejected_visited_components:
                        rejected_visited_components.add(key)
                        self.corridor_sweep_history.append({
                            "event":
                                "VISITED_SIDE_COMPONENT_GOAL_REJECTED",
                            "elapsed_sec": round(self.elapsed(), 3),
                            "candidate_target":
                                [float(target[0]), float(target[1])],
                            "frontier_center":
                                [float(frontier_point[0]),
                                 float(frontier_point[1])],
                            "side": stable_side,
                            "component_cells":
                                component.get("component_cells"),
                            "matched_branch_ids":
                                component.get("matched_branch_ids", []),
                        })
                    continue
                yaw = math.atan2(float(side * normal[1]),
                                 float(side * normal[0]))
                score_item = score_by_id.get(
                    int(candidate.get("id", -1)), {})
                branch = self.branch_scheduler.register(
                    station_coordinate, stable_side,
                    (float(target[0]), float(target[1])), self.elapsed(),
                    (float(frontier_point[0]), float(frontier_point[1])),
                    entry_quality=depth)
                # Registration can merge several frontier observations at one
                # doorway station.  The scheduler intentionally retains the
                # deepest previously verified endpoint, so execution must use
                # that persistent endpoint as well.  Returning the latest
                # shallow observation here made the queue status say 2.1 m
                # while the robot was actually sent to a different door-edge
                # point.
                execution_target = branch.entry_target
                execution_depth = branch.entry_quality
                endpoint_distance = math.hypot(
                    float(execution_target[0]) - float(pose[0]),
                    float(execution_target[1]) - float(pose[1]))
                if endpoint_distance < max(
                        0.45, self.reached_tolerance + 0.15):
                    self.branch_scheduler.mark_entry_failed(
                        branch.branch_id, self.elapsed())
                    self.corridor_sweep_history.append({
                        "event": "CORRIDOR_SIDE_NEAR_ENDPOINT_REJECTED",
                        "elapsed_sec": round(self.elapsed(), 3),
                        "branch_id": branch.branch_id,
                        "endpoint": [float(execution_target[0]),
                                     float(execution_target[1])],
                        "distance_m": endpoint_distance,
                        "reason": "already_at_endpoint_is_not_entry",
                    })
                    continue
                qualified.append((branch.branch_id, {
                    "position": [float(execution_target[0]),
                                 float(execution_target[1]),
                                 float(pose[2])],
                    "yaw": yaw,
                    "orientation": [0.0, 0.0, math.sin(0.5 * yaw),
                                    math.cos(0.5 * yaw)],
                    "candidate_id": candidate.get("id"),
                    "frontier_id": candidate.get("frontier_id"),
                    "frontier_center": frontier[:2],
                    "frontier_size": candidate.get("frontier_size"),
                    "raw_frontier_evidence": bool(
                        candidate.get("raw_frontier_evidence", False)),
                    "information_gain": score_item.get(
                        "information_gain", 0.0),
                    "score": score_item.get("score"),
                    "source": "corridor_side_coverage_debt",
                    "scheduler_phase": "SIDE_REGION_ENTRY",
                    "physical_entry_target_depth_m": execution_depth,
                    "staging_entry_target":
                        execution_depth < 1.45 - 1e-6,
                    "side_frontier_lateral_m": abs(signed_lateral),
                    "side_frontier_longitudinal_m": longitudinal,
                    "side_frontier_longitudinal_abs_m":
                        abs(longitudinal),
                    "corridor_branch_id": branch.branch_id,
                    "corridor_branch_station_m": branch.station,
                    "corridor_branch_side": branch.side,
                    "corridor_branch_state": branch.state,
                }))
        discovered = {}
        for branch_id, goal in qualified:
            branch = self.branch_scheduler.find(branch_id)
            if branch is not None:
                discovered[branch_id] = {
                    "branch_id": branch.branch_id,
                    "station_m": branch.station,
                    "side": branch.side,
                    "entry_target": list(branch.entry_target),
                    "entry_depth_m": branch.entry_quality,
                    "staging_only": branch.entry_quality < 1.45 - 1e-6,
                    "observations": branch.observations,
                }
        if discovered:
            self.corridor_sweep_history.append({
                "event": "CORRIDOR_SIDE_BRANCH_QUEUE_UPDATED",
                "elapsed_sec": round(self.elapsed(), 3),
                "branches": list(discovered.values()),
                "both_sides_present":
                    len({item["side"] for item in discovered.values()}) > 1,
            })
        if not qualified:
            return None
        if not self.enable_corridor_branch_scheduler:
            return qualified[0][1]
        selected = self.branch_scheduler.select(
            [item[0] for item in qualified],
            self._corridor_station((pose[0], pose[1])), self.elapsed())
        if selected is None:
            return None
        goal = next(item[1] for item in qualified
                    if item[0] == selected.branch_id)
        goal["scheduler_phase"] = (
            "PAIRED_SIDE_REGION_ENTRY"
            if self.branch_scheduler.active_station is not None and
            abs(selected.station - self.branch_scheduler.active_station) <=
            self.corridor_branch_station_tolerance else
            "SIDE_REGION_ENTRY")
        return goal

    def _remembered_unvisited_corridor_branch_goal(self, cycle, pose, grid):
        """Return a reachable, discovered side branch without reopening FUEL.

        This is the only permitted rearward exploration endpoint while the
        mission is not returning home: the endpoint must be a persistent
        UNENTERED lateral branch, never an old corridor-centre goal.
        """
        current_station = self._corridor_station((pose[0], pose[1]))
        candidate_ids = [
            branch.branch_id for branch in self.branch_scheduler.branches
            if (branch.state == "UNENTERED" and
                self._rearward_branch_is_paired_opposite(
                    branch.station, branch.side, current_station))]
        while candidate_ids:
            branch = self.branch_scheduler.select(
                candidate_ids,
                self._corridor_station((pose[0], pose[1])), self.elapsed())
            if branch is None:
                return None
            result = astar_safe_path(
                grid, (float(pose[0]), float(pose[1])), branch.entry_target,
                self.clearance, self.reached_tolerance,
                allow_blocked_start=True)
            if result.get("success"):
                dx = branch.entry_target[0] - float(pose[0])
                dy = branch.entry_target[1] - float(pose[1])
                yaw = math.atan2(dy, dx)
                goal = {
                    "position": [branch.entry_target[0],
                                 branch.entry_target[1], float(pose[2])],
                    "yaw": yaw,
                    "orientation": [0.0, 0.0, math.sin(0.5 * yaw),
                                    math.cos(0.5 * yaw)],
                    "information_gain": 0.0,
                    "score": 0.0,
                    "source": "corridor_remembered_side_branch",
                    "scheduler_phase": "REMEMBERED_SIDE_REGION_ENTRY",
                    "corridor_branch_id": branch.branch_id,
                    "corridor_branch_station_m": branch.station,
                    "corridor_branch_side": branch.side,
                    "corridor_branch_state": branch.state,
                    "_preplanned_path_result": self._plan_path(cycle, {
                        "position": [branch.entry_target[0],
                                     branch.entry_target[1], float(pose[2])],
                    }),
                }
                self.corridor_sweep_history.append({
                    "event": "REMEMBERED_UNVISITED_SIDE_BRANCH_SELECTED",
                    "elapsed_sec": round(self.elapsed(), 3),
                    "branch_id": branch.branch_id,
                    "branch_station_m": branch.station,
                    "branch_side": branch.side,
                    "goal": list(branch.entry_target),
                })
                return goal
            candidate_ids.remove(branch.branch_id)
        return None

    def _active_side_branch_continuation_goal(self, cycle, pose, grid):
        """Deepen one entered side branch or return to its corridor station.

        A shallow, safely reachable staging endpoint is useful for growing an
        oblique doorway scan, but it must not hand control straight back to a
        global FUEL goal.  Otherwise FUEL can select the opposite room's high
        gain frontier and drive one segment across room -> corridor -> room;
        the doorway recognizer then assigns the wrong portal orientation and
        cannot return to the corridor.  Continue along the same online branch
        until room-entry depth is available.  If it is not, return to the
        branch's remembered corridor station before considering another side.
        """
        branch = self.branch_scheduler.find(
            self.branch_scheduler.active_branch_id)
        if (branch is None or self.active_region_side_sign is None or
                self.room_scheduler.active_door is not None or grid is None or
                self.corridor_station_axis is None or
                self.corridor_station_origin is None):
            return None
        axis = np.asarray(self.corridor_station_axis, dtype=float)
        axis /= max(1e-9, float(np.linalg.norm(axis)))
        normal = np.asarray([-axis[1], axis[0]], dtype=float)
        station_origin = np.asarray(
            self.corridor_station_origin, dtype=float)
        corridor_station = station_origin + branch.station * axis
        current = np.asarray([float(pose[0]), float(pose[1])], dtype=float)
        current_depth = branch.side * float(np.dot(
            current - corridor_station, normal))
        required_depth = max(self.corridor_side_entry_depth, 2.0)
        # Once the robot has reached room-confirming depth, give the crossing
        # detector one normal planning cycle.  If it still has no valid door,
        # fall through to the corridor return instead of crossing both sides.
        if current_depth < required_depth - 0.20:
            offset_count = int(math.floor(
                self.corridor_side_station_search_radius /
                self.corridor_side_station_search_step + 1e-9))
            station_offsets = [0.0]
            for index in range(1, offset_count + 1):
                delta = index * self.corridor_side_station_search_step
                station_offsets.extend((delta, -delta))
            for depth in (required_depth,
                          max(current_depth + 0.45, 1.75)):
                if depth <= current_depth + 0.20:
                    continue
                for station_offset in station_offsets:
                    target_station = corridor_station + station_offset * axis
                    target = (target_station +
                              float(branch.side) * depth * normal)
                    cell = grid.world_to_cell(
                        (float(target[0]), float(target[1])))
                    if (cell is None or
                            int(grid.data[cell[1], cell[0]]) != 0):
                        continue
                    path = astar_safe_path(
                        grid, (float(pose[0]), float(pose[1])),
                        (float(target[0]), float(target[1])),
                        self.clearance, self.reached_tolerance,
                        allow_blocked_start=True)
                    if not path.get("success"):
                        continue
                    branch = self.branch_scheduler.register(
                        branch.station + station_offset, branch.side,
                        (float(target[0]), float(target[1])), self.elapsed(),
                        branch.frontier, entry_quality=depth)
                    yaw = math.atan2(
                        float(branch.side * normal[1]),
                        float(branch.side * normal[0]))
                    goal = {
                        "position": [float(target[0]), float(target[1]),
                                     float(pose[2])],
                        "yaw": yaw,
                        "orientation": [0.0, 0.0, math.sin(0.5 * yaw),
                                        math.cos(0.5 * yaw)],
                        "information_gain": 0.0,
                        "score": 0.0,
                        "source": "corridor_side_branch_deepen",
                        "scheduler_phase": "SIDE_REGION_DEEPEN",
                        "corridor_branch_id": branch.branch_id,
                        "corridor_branch_station_m": branch.station,
                        "corridor_branch_side": branch.side,
                        "physical_entry_target_depth_m": depth,
                        "_preplanned_path_result": self._plan_path(cycle, {
                            "position": [float(target[0]), float(target[1]),
                                         float(pose[2])],
                        }),
                    }
                    self.corridor_sweep_history.append({
                        "event": "CORRIDOR_SIDE_BRANCH_DEEPEN_SELECTED",
                        "elapsed_sec": round(self.elapsed(), 3),
                        "branch_id": branch.branch_id,
                        "side": branch.side,
                        "from_depth_m": current_depth,
                        "target_depth_m": depth,
                        "goal": [float(target[0]), float(target[1])],
                    })
                    return goal
        return_goal = self._region_return_goal(
            (float(corridor_station[0]), float(corridor_station[1])),
            pose, grid, "side_region_corridor_return")
        if return_goal is not None:
            return_goal.update({
                "scheduler_phase": "SIDE_REGION_CORRIDOR_RETURN",
                "corridor_branch_id": branch.branch_id,
                "corridor_branch_station_m": branch.station,
                "corridor_branch_side": branch.side,
                "cross_side_goal_guard": True,
            })
            self.corridor_sweep_history.append({
                "event": "CROSS_SIDE_FUEL_GUARD_RETURN_SELECTED",
                "elapsed_sec": round(self.elapsed(), 3),
                "branch_id": branch.branch_id,
                "side": branch.side,
                "current_depth_m": current_depth,
                "corridor_goal": [float(corridor_station[0]),
                                  float(corridor_station[1])],
            })
        return return_goal

    def _region_return_goal(self, entry, pose, grid, source,
                            coverage=None, expanded=False):
        """Build a safe return near a saved entry, relocating it if necessary."""
        radius = (self.region_exit_recovery_radius if expanded else
                  self.region_return_relocation_radius)
        preferred = []
        axis = self.corridor_station_axis
        if axis is None:
            axis = self.corridor_axis
        if axis is not None:
            axis = np.asarray(axis, dtype=float)
            norm = float(np.linalg.norm(axis))
            if norm > 1e-9:
                axis /= norm
                # A historical entry can lie on an inflated wall pixel.  Probe
                # both directions along the established corridor centreline.
                for distance in (0.30, -0.30, 0.60, -0.60,
                                 0.90, -0.90, 1.20, -1.20):
                    preferred.append((
                        float(entry[0]) + distance * float(axis[0]),
                        float(entry[1]) + distance * float(axis[1])))
        selection = nearest_reachable_goal(
            grid, (float(pose[0]), float(pose[1])),
            (float(entry[0]), float(entry[1])),
            min(self.clearance, 0.20), self.reached_tolerance,
            radius, 96 if expanded else 64, preferred,
            self.region_return_failed_targets)
        event = {
            "event": ("REGION_EXIT_RECOVERY_SEARCH" if expanded else
                      "REGION_RETURN_RELOCATION_SEARCH"),
            "elapsed_sec": round(self.elapsed(), 3),
            "original_entry": [float(entry[0]), float(entry[1])],
            "search_radius_m": radius,
            "success": bool(selection.get("success")),
            "reason": selection.get("reason"),
            "selected_target": selection.get("target"),
            "relocation_distance_m": selection.get("displacement"),
            "candidates_tested": selection.get("candidates_tested", 0),
            "excluded_failed_targets":
                len(self.region_return_failed_targets),
        }
        self.depth_breadth_history.append(event)
        if not selection.get("success"):
            return None
        target = selection["target"]
        displacement = float(selection.get("displacement", 0.0))
        return {
            "position": [float(target[0]), float(target[1]), pose[2]],
            "information_gain": 0.0,
            "score": 0.0 if coverage is not None else None,
            "source": source,
            "scheduler_phase": ("REGION_EXIT_RECOVERY" if expanded else
                                "REGION_RETURN"),
            "region_entry_return": True,
            "region_coverage": coverage,
            "planning_clearance_m": min(self.clearance, 0.20),
            "region_return_direct_astar": True,
            "region_entry_original":
                [float(entry[0]), float(entry[1])],
            "region_entry_relocated": displacement > 1e-6,
            "region_return_relocation_distance_m": displacement,
            "region_return_search_radius_m": radius,
            "region_return_candidates_tested":
                int(selection.get("candidates_tested", 0)),
        }

    def _plan_region_exit_recovery(self, cycle):
        """Use a wider safe-goal search before declaring frontier exhaustion."""
        if (not self.depth_breadth.entry_observed or
                self.region_exit_recovery_attempts >=
                self.region_exit_recovery_limit):
            return None
        with self.lock:
            pose, grid = self.pose, self.grid
        entry = (self.depth_breadth.physical_entry_anchor or
                 self.depth_breadth.entry_anchor)
        if pose is None or grid is None or entry is None:
            return None
        goal = self._region_return_goal(
            entry, pose, grid, "region_exit_recovery",
            self.depth_breadth.coverage_status, expanded=True)
        if goal is not None:
            self.region_exit_recovery_attempts += 1
            goal["region_exit_recovery_attempt"] = \
                self.region_exit_recovery_attempts
        return goal

    def _plan_corridor_monotonic_forward_probe(self, cycle, pose, grid, axis):
        """Advance one F2-only, densely 3-D-checked centreline segment.

        A stale/inflated 2-D projection can reject every A* target even while
        the independent 3-D body service observes a clear physical corridor.
        This fallback never crosses a reported occupied footprint, never runs
        with an active room, and never runs after a forward terminal wall is
        observed.  F1 cannot enter it because its enable parameter defaults
        to false and is overridden only inside the fresh F2 process.
        """
        if (not self.enable_corridor_monotonic_forward_probe or
                pose is None or grid is None or axis is None or
                self.room_scheduler.active_door is not None or
                self._missing_room_retrace_active() or
                self.corridor_monotonic_probe_count >=
                self.corridor_monotonic_probe_limit):
            return None, None
        exited_rooms = sum(bool(door.visited) for door in
                           self.room_scheduler.detector.doors)
        if exited_rooms >= self.room_target_count:
            return None, None
        direction = np.asarray(axis, dtype=float)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            return None, None
        direction /= norm
        if self._forward_corridor_wall_observed(
                pose, grid, direction,
                maximum_distance=self.corridor_monotonic_probe_distance + .35):
            self.corridor_monotonic_probe_mode_active = False
            self.corridor_sweep_history.append({
                "event": "CORRIDOR_MONOTONIC_PROBE_WITHHELD_TERMINAL_WALL",
                "elapsed_sec": round(self.elapsed(), 3),
                "probe_count": self.corridor_monotonic_probe_count,
            })
            return None, None

        line_axis = (np.asarray(self.corridor_station_axis, dtype=float)
                     if self.corridor_station_axis is not None else direction)
        line_norm = float(np.linalg.norm(line_axis))
        if line_norm < 1e-6:
            return None, None
        line_axis /= line_norm
        if float(np.dot(line_axis, direction)) < 0.0:
            line_axis = -line_axis
        origin = (np.asarray(self.corridor_station_origin, dtype=float)
                  if self.corridor_station_origin is not None else
                  np.asarray(self.corridor_anchor, dtype=float)
                  if self.corridor_anchor is not None else
                  np.asarray(pose[:2], dtype=float))
        current = np.asarray(pose[:2], dtype=float)
        centerline = origin + np.dot(current-origin, line_axis)*line_axis
        target = centerline + self.corridor_monotonic_probe_distance*line_axis
        delta = target-current
        distance = float(np.linalg.norm(delta))
        if distance <= self.reached_tolerance + .05:
            return None, None
        sample_count = max(2, int(math.ceil(
            distance/self.corridor_monotonic_probe_sample_spacing)))
        heading = math.atan2(float(delta[1]), float(delta[0]))
        checked = []
        for index in range(1, sample_count+1):
            fraction = float(index)/float(sample_count)
            point = current + fraction*delta
            footprint = self._scan_check(
                float(point[0]), float(point[1]), heading)
            checked.append((float(point[0]), float(point[1])))
            if (not footprint.map_available or
                    footprint.occupied_collision or
                    int(footprint.unknown_queries) >
                    self.corridor_monotonic_probe_max_unknown_queries):
                self.corridor_sweep_history.append({
                    "event": "CORRIDOR_MONOTONIC_PROBE_3D_REJECTED",
                    "elapsed_sec": round(self.elapsed(), 3),
                    "sample_index": index,
                    "sample_count": sample_count,
                    "sample": [round(float(point[0]), 4),
                               round(float(point[1]), 4)],
                    "map_available": bool(footprint.map_available),
                    "occupied_collision": bool(
                        footprint.occupied_collision),
                    "unknown_queries": int(footprint.unknown_queries),
                    "status": footprint.status,
                })
                return None, None
        self.corridor_monotonic_probe_count += 1
        self.corridor_monotonic_probe_mode_active = True
        path = [(float(current[0]), float(current[1]))] + checked
        goal = {
            "position": [float(target[0]), float(target[1]), float(pose[2])],
            "information_gain": 0.0,
            "score": 0.0,
            "source": "corridor_monotonic_forward_probe",
            "scheduler_phase": "CORRIDOR_SWEEP",
            "corridor_advance_m": self.corridor_monotonic_probe_distance,
            "corridor_monotonic_probe": True,
            "verified_corridor_3d_probe": True,
            "execution_timeout_sec": 10.0,
            "_preplanned_path_result": {
                "success": True,
                "reason": "dense_3d_verified_corridor_monotonic_probe",
                "path": path,
                "execution_waypoints": [checked[-1]],
            },
        }
        self.corridor_sweep_history.append({
            "event": "CORRIDOR_MONOTONIC_FORWARD_PROBE_SELECTED",
            "elapsed_sec": round(self.elapsed(), 3),
            "probe_count": self.corridor_monotonic_probe_count,
            "sample_count": sample_count,
            "start": [float(current[0]), float(current[1])],
            "target": [float(target[0]), float(target[1])],
            "axis": [float(line_axis[0]), float(line_axis[1])],
        })
        return goal, goal["_preplanned_path_result"]

    def _plan_corridor_no_frontier_recovery(self, cycle):
        """Continue the corridor before accepting global frontier exhaustion."""
        with self.lock:
            pose, grid = self.pose, self.grid
        if pose is None or grid is None:
            return None
        # A missing-room pass is an observation window around the corridor
        # terminal, never a route back to the lobby.  Once its fixed-frame
        # distance is consumed, restore the original outbound axis before
        # asking for another corridor waypoint.
        self._finish_bounded_missing_room_retrace(pose)
        context = self._corridor_context(pose, grid)
        # Before the full corridor latch, two consecutive raw geometric
        # confirmations are sufficient for a forward transit probe.  This
        # bypasses repeated short FUEL goals in the lobby, but never enables
        # side-door takeover (that remains gated by corridor_established).
        axis = (np.asarray(self.corridor_axis, dtype=float)
                if self.corridor_axis is not None else
                np.asarray(context["axis"], dtype=float)
                if context is not None and
                (context.get("is_corridor") or
                 (context.get("raw_is_corridor") and
                  int(context.get("confirmation_count", 0)) >= 1)) else None)
        if axis is None:
            return None
        retrace_remaining = None
        if (self._missing_room_retrace_active() and
                self.terminal_missing_room_retrace_start_station is not None and
                self.corridor_station_axis is not None):
            sign = (self.corridor_forward_station_sign
                    if self.corridor_forward_station_sign is not None else 1.0)
            directed = float(sign) * self._corridor_station(pose)
            travelled = max(
                0.0,
                float(self.terminal_missing_room_retrace_start_station) -
                directed)
            retrace_remaining = max(
                0.0,
                self.terminal_missing_room_retrace_max_distance - travelled)
        replacement, replacement_path = self._corridor_forward_plan(
            cycle, pose, grid, axis, maximum_advance=retrace_remaining)
        event = "CORRIDOR_NO_FRONTIER_FORWARD"
        if replacement is None:
            self.corridor_no_frontier_exhaustion_streak += 1
        else:
            self.corridor_no_frontier_exhaustion_streak = 0
            self.corridor_monotonic_probe_mode_active = False
            self.last_corridor_forward_success_at = self.elapsed()
        # A stationary rescan cannot expose cells hidden just beyond the
        # current LiDAR horizon. After two failed ordinary probes, advance a
        # single body length on the latched centreline. A* still requires this
        # entire short segment to be observed free, so no unknown cell or
        # terminal wall is crossed.
        if (replacement is None and
                self.corridor_no_frontier_exhaustion_streak >=
                self.corridor_no_frontier_probe_after and
                not (self.room_scheduler.active_door is None and
                     sum(bool(door.visited) for door in
                         self.room_scheduler.detector.doors) >=
                     self.room_target_count) and
                self.corridor_no_frontier_probe_distance >
                self.reached_tolerance + 0.10):
            replacement, replacement_path = self._corridor_forward_plan(
                cycle, pose, grid, axis,
                maximum_advance=self.corridor_no_frontier_probe_distance)
            if replacement is not None:
                event = "CORRIDOR_NO_FRONTIER_SHORT_PROBE"
                self.corridor_no_frontier_exhaustion_streak = 0
        # If the 2-D projection remains disconnected, do not rotate forever.
        # F2 may use one short segment whose complete swept body envelope was
        # independently checked in 3-D. Once the mode has made progress, the
        # next segment can be selected without paying for another stationary
        # rescan; every segment is still freshly collision checked.
        monotonic_after = (1 if self.corridor_monotonic_probe_mode_active
                           else self.corridor_monotonic_probe_after)
        if (replacement is None and
                self.corridor_no_frontier_exhaustion_streak >=
                monotonic_after):
            replacement, replacement_path = \
                self._plan_corridor_monotonic_forward_probe(
                    cycle, pose, grid, axis)
            if replacement is not None:
                event = "CORRIDOR_MONOTONIC_FORWARD_PROBE"
                self.corridor_no_frontier_exhaustion_streak = 0
        # A missing forward ray during initial corridor mapping is normally
        # caused by limited map horizon, not a terminal wall.  Reversing here
        # makes the lobby a tempting recovery target.  Permit a global
        # reversal only after all required rooms have been exited; before that
        # keep the established forward axis and ask the next map update for a
        # forward path.
        all_rooms_exited = (
            self.room_scheduler.active_door is None and
            sum(bool(door.visited)
                for door in self.room_scheduler.detector.doors) >=
            self.room_target_count)
        if (replacement is None and all_rooms_exited and
                self.corridor_reversal_count < self.corridor_reversal_limit):
            axis = -axis
            self.corridor_axis = axis
            self.corridor_reversed = True
            self.corridor_reversal_count += 1
            replacement, replacement_path = self._corridor_forward_plan(
                cycle, pose, grid, axis)
            event = "CORRIDOR_NO_FRONTIER_GLOBAL_REVERSED"
        self.corridor_sweep_history.append({
            "event": event if replacement is not None else
                     "CORRIDOR_NO_FRONTIER_EXHAUSTED",
            "elapsed_sec": round(self.elapsed(), 3),
            "replacement_goal": (replacement.get("position")
                                 if replacement is not None else None),
            "axis": [float(axis[0]), float(axis[1])],
            "exhaustion_streak": self.corridor_no_frontier_exhaustion_streak,
        })
        if replacement is not None and replacement_path is not None:
            replacement["_preplanned_path_result"] = replacement_path
        return replacement

    def _forward_corridor_wall_observed(self, pose, grid, axis,
                                        maximum_distance=None):
        """Return true only when occupied cells, not unknown cells, stop ahead."""
        if pose is None or grid is None or axis is None:
            return False
        direction = np.asarray(axis, dtype=float)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            return False
        direction /= norm
        lateral = np.asarray([-direction[1], direction[0]])
        if maximum_distance is None:
            max_distance = max(5.0, float(self.room_exit_terminal_wall_distance))
        else:
            try:
                max_distance = max(0.5, float(maximum_distance))
            except (TypeError, ValueError):
                return False
        for distance in np.arange(0.5, max_distance + 0.1,
                                  max(0.20, float(grid.resolution))):
            occupied_at_distance = 0
            for offset in (-0.35, 0.0, 0.35):
                point = (float(pose[0]) + distance * direction[0] + offset * lateral[0],
                         float(pose[1]) + distance * direction[1] + offset * lateral[1])
                cell = grid.world_to_cell(point)
                if cell is None:
                    continue
                ix, iy = int(cell[0]), int(cell[1])
                if 0 <= ix < grid.width and 0 <= iy < grid.height:
                    # Occupied is explicitly known occupied. Unknown (=-1)
                    # must not be mistaken for a terminal wall.
                    if int(grid.data[iy, ix]) >= 50:
                        occupied_at_distance += 1
            # A terminal wall is a transverse band at one longitudinal
            # station.  Two unrelated obstacle returns at different ranges
            # must not accumulate into a false corridor end.
            # Require the complete narrow cross-section.  A centre return
            # plus one piece of furniture/noise beside it was still enough to
            # create false terminals in run33 at x~=20 and later x~=16.
            if occupied_at_distance >= 3:
                return True
        return False

    def _plan_post_room_corridor_sweep_goal(self, cycle):
        """Probe forward after room coverage before committing to return-home."""
        if (not self.enable_post_room_corridor_sweep or
                self.post_room_corridor_sweep_done):
            return None
        with self.lock:
            pose, grid = self.pose, self.grid
        if pose is None or grid is None:
            return None
        context = self._corridor_context(pose, grid)
        if not context or not context.get("is_corridor"):
            return None
        axis = (self.corridor_axis if self.corridor_axis is not None
                else context.get("axis"))
        if axis is None:
            return None
        # A goal-count limit is only a re-planning guard.  It must never be
        # interpreted as reaching the corridor end, otherwise the manager can
        # return home while a large unexplored forward section remains.
        if (self.post_room_corridor_sweep_attempts >=
                self.post_room_corridor_sweep_max_goals):
            wall = self._forward_corridor_wall_observed(pose, grid, axis)
            self.post_room_corridor_sweep_done = bool(wall)
            self.corridor_sweep_history.append({
                "event": "POST_ROOM_CORRIDOR_WALL_GATE" if wall else
                         "POST_ROOM_CORRIDOR_WAITING_FOR_WALL",
                "elapsed_sec": round(self.elapsed(), 3),
                "wall_observed": bool(wall),
                "attempt": self.post_room_corridor_sweep_attempts,
            })
            return None
        replacement, replacement_path = self._corridor_forward_plan(
            cycle, pose, grid, axis)
        if replacement is not None and replacement_path is not None:
            self.post_room_corridor_sweep_attempts += 1
            replacement["source"] = "post_room_corridor_sweep"
            replacement["scheduler_phase"] = "POST_ROOM_CORRIDOR_SWEEP"
            replacement["post_room_corridor_sweep_attempt"] = \
                self.post_room_corridor_sweep_attempts
            replacement["_preplanned_path_result"] = replacement_path
            self.corridor_sweep_history.append({
                "event": "POST_ROOM_CORRIDOR_SWEEP_GOAL",
                "elapsed_sec": round(self.elapsed(), 3),
                "attempt": self.post_room_corridor_sweep_attempts,
                "goal": replacement.get("position"),
                "axis": [float(axis[0]), float(axis[1])],
            })
            return replacement
        wall = self._forward_corridor_wall_observed(pose, grid, axis)
        self.post_room_corridor_sweep_done = bool(wall)
        self.corridor_sweep_history.append({
            "event": "POST_ROOM_CORRIDOR_TERMINAL_WALL" if wall else
                     "POST_ROOM_CORRIDOR_SWEEP_EXHAUSTED",
            "elapsed_sec": round(self.elapsed(), 3),
            "wall_observed": bool(wall),
            "axis": [float(axis[0]), float(axis[1])],
        })
        return None

    def _apply_corridor_sweep(self, cycle, goal, path_result):
        """Reject short/backtracking generic targets in a recognized corridor."""
        # Corridor forcing existed to give the doorway recognizer repeated,
        # orderly side views.  In door-free region mode it suppresses exactly
        # the lateral FUEL frontiers needed to cover rooms, causing long
        # corridor oscillations.  Preserve FUEL's selected reachable frontier.
        if not self.room_scheduler.config.enabled:
            return goal, path_result
        source = str(goal.get("source", ""))
        if (not path_result.get("success") or
                str(goal.get("source", "")).startswith("lightweight_room_") or
                source in ("mission_return_home",
                           # This is an intentional return transit from the
                           # terminal wall to the latched corridor entrance,
                           # never a backward exploration proposal.
                           "stair_corridor_exit_handoff",
                           "post_room_corridor_sweep",
                           "corridor_side_coverage_debt",
                           "corridor_side_branch_deepen",
                           "corridor_remembered_side_branch",
                           "corridor_resume_centerline",
                           "corridor_door_forward_commit",
                           "corridor_monotonic_forward_probe",
                           # A paired opposite doorway is a short, verified
                           # recentering manoeuvre.  Do not replace it with a
                           # five-metre forward sweep before the doorway
                           # scheduler has had the chance to commit it.
                           "paired_opposite_door_recenter")):
            if source in ("corridor_side_coverage_debt",
                           "corridor_remembered_side_branch"):
                self.corridor_sweep_history.append({
                    "event": "CORRIDOR_SIDE_GOAL_PRESERVED",
                    "elapsed_sec": round(self.elapsed(), 3),
                    "source": source,
                    "goal": goal.get("position"),
                    "branch_id": goal.get("corridor_branch_id"),
                })
            return goal, path_result
        with self.lock:
            pose, grid = self.pose, self.grid
            trajectory = [(item[1], item[2]) for item in self.trajectory]
        context = self._corridor_context(pose, grid)
        if context is None or not context["is_corridor"]:
            return goal, path_result
        point = goal.get("position") or []
        if len(point) < 2:
            return goal, path_result
        axis = context["axis"]
        # Generic corridor progress goals should use the corridor centre, not
        # a wall-hugging FUEL endpoint.  Side-door and room goals are handled
        # separately and must retain their lateral entry geometry.
        centre_sources = {
            "fuel_frontier", "corridor_sweep", "corridor_forward_no_frontier",
            "corridor_no_frontier_recovery", "corridor_forward_recovery",
        }
        if (source in centre_sources and
                (self.corridor_station_origin is not None or
                 self.corridor_anchor is not None)):
            raw = np.asarray([float(point[0]), float(point[1])], dtype=float)
            anchor = np.asarray(
                self.corridor_station_origin
                if self.corridor_station_origin is not None else
                self.corridor_anchor, dtype=float)
            centre_axis = np.asarray(
                self.corridor_station_axis
                if self.corridor_station_axis is not None else axis,
                dtype=float)
            centre_axis /= max(1e-9, float(np.linalg.norm(centre_axis)))
            if float(np.dot(centre_axis, axis)) < 0.0:
                centre_axis = -centre_axis
            centre = anchor + np.dot(raw - anchor, centre_axis) * centre_axis
            lateral = float(np.linalg.norm(raw - centre))
            if lateral > 0.45:
                # Move only the lateral component; preserve forward station.
                shifted = centre
                cell = grid.world_to_cell((float(shifted[0]), float(shifted[1])))
                if (cell is not None and int(grid.data[cell[1], cell[0]]) == 0):
                    shifted_goal = dict(goal)
                    shifted_goal["position"] = [float(shifted[0]),
                                                  float(shifted[1]),
                                                  float(point[2]) if len(point) > 2 else float(pose[2])]
                    shifted_path = self._plan_path(cycle, shifted_goal)
                    if shifted_path.get("success"):
                        self.corridor_sweep_history.append({
                            "event": "CORRIDOR_GOAL_CENTERLINE_SHIFT",
                            "elapsed_sec": round(self.elapsed(), 3),
                            "source": source,
                            "original_goal": list(point),
                            "shifted_goal": list(shifted_goal["position"]),
                            "lateral_error_m": lateral,
                        })
                        goal, path_result = shifted_goal, shifted_path
                        point = goal["position"]
        progress = ((float(point[0]) - pose[0]) * float(axis[0]) +
                    (float(point[1]) - pose[1]) * float(axis[1]))
        if not self.corridor_direction_aligned:
            # Heading can initially point opposite the first meaningful FUEL
            # advance.  This is sign alignment, not the one allowed global
            # reverse sweep.
            if progress < 0.0:
                axis = -np.asarray(axis, dtype=float)
                self.corridor_axis = np.asarray(axis, dtype=float)
                progress = -progress
            self.corridor_direction_aligned = True
        recent_distance = min((math.hypot(float(point[0]) - old[0],
                                          float(point[1]) - old[1])
                               for old in trajectory[-300:]), default=math.inf)
        visited_room = self.room_scheduler.goal_in_visited_room(
            (float(point[0]), float(point[1])))
        lateral_error = 0.0
        if (self.corridor_station_origin is not None and
                self.corridor_station_axis is not None):
            lateral_error = corridor_line_lateral_error(
                point, self.corridor_station_origin,
                self.corridor_station_axis)
        elif self.corridor_anchor is not None:
            offset = np.asarray([float(point[0]), float(point[1])]) - np.asarray(
                self.corridor_anchor, dtype=float)
            lateral_error = abs(float(axis[0]) * float(offset[1]) -
                                float(axis[1]) * float(offset[0]))
        if (self.corridor_minimum_advance <= progress <=
                self.corridor_forward_distance + 0.25 and
                lateral_error <= 0.60 and
                recent_distance >= self.corridor_recent_exclusion and
                not visited_room):
            return goal, path_result
        replacement, replacement_path = self._corridor_forward_plan(
            cycle, pose, grid, axis)
        event = "CORRIDOR_FORWARD_REPLACED"
        if (replacement is None and
                self._mission_rooms_physically_exited() and
                self.corridor_reversal_count < self.corridor_reversal_limit):
            self.corridor_axis = -np.asarray(axis)
            self.corridor_reversed = True
            self.corridor_reversal_count += 1
            replacement, replacement_path = self._corridor_forward_plan(
                cycle, pose, grid, self.corridor_axis)
            event = "CORRIDOR_GLOBAL_REVERSED"
        self.corridor_sweep_history.append({
            "event": event if replacement is not None else
                     "CORRIDOR_BACKWARD_GOAL_REJECTED",
            "elapsed_sec": round(self.elapsed(), 3),
            "original_goal": point, "original_progress_m": progress,
            "recent_distance_m": recent_distance,
            "inside_visited_room": visited_room,
            "corridor_centerline_error_m": lateral_error,
            "replacement_goal": replacement.get("position") if replacement else None,
            "axis": [float(axis[0]), float(axis[1])],
        })
        if replacement is not None:
            return replacement, replacement_path
        remembered = self._remembered_unvisited_corridor_branch_goal(
            cycle, pose, grid)
        if remembered is not None:
            remembered_path = remembered.pop("_preplanned_path_result", None)
            return remembered, remembered_path
        rejected = dict(path_result)
        rejected.update({
            "success": False,
            "reason": "corridor_backward_exploration_goal_rejected",
            "path": [],
        })
        return goal, rejected

    def _plan_fuel_goal(self, cycle):
        self._set_state("PLAN_EXPLORATION_GOAL")
        with self.lock:
            pose = self.pose
            frame = self.pose_frame
            grid = self.grid
        corridor = self._corridor_context(pose, grid)
        in_corridor = self._on_established_corridor_centerline(
            pose, corridor)
        # A wide room can itself satisfy the local aspect-ratio corridor test.
        # Once a remembered side-branch endpoint has actually been reached,
        # branch commitment is stronger evidence than that transient local
        # classification.  Continue/deepen that same branch (or return to its
        # corridor station) instead of launching a long corridor-sweep goal
        # from inside the room.
        if self.active_region_side_sign is not None:
            continuation = self._active_side_branch_continuation_goal(
                cycle, pose, grid)
            if continuation is not None:
                preplanned = continuation.pop("_preplanned_path_result", None)
                if preplanned is not None:
                    continuation["_preplanned_path_result"] = preplanned
                if self.first_goal_elapsed is None:
                    self.first_goal_elapsed = self.elapsed()
                return continuation, {
                    "source": continuation.get("source"),
                    "corridor_branch_id": continuation.get(
                        "corridor_branch_id"),
                    "cross_side_goal_guard": True,
                }, 0
        if in_corridor:
            remembered = self._remembered_unvisited_corridor_branch_goal(
                cycle, pose, grid)
            if remembered is not None:
                if self.first_goal_elapsed is None:
                    self.first_goal_elapsed = self.elapsed()
                return remembered, {
                    "source": remembered.get("source"),
                    "corridor_branch_id": remembered.get(
                        "corridor_branch_id"),
                }, 0
        self.depth_breadth.ensure_started((pose[0], pose[1]), self.elapsed())
        exclusion = None if in_corridor else self.depth_breadth.exclusion(self.elapsed())
        if (exclusion is not None and
                self.depth_breadth.entry_return_pending and
                exclusion.get("entry_anchor") is not None):
            entry = exclusion["entry_anchor"]
            if math.hypot(entry[0] - pose[0], entry[1] - pose[1]) <= \
                    self.reached_tolerance:
                self.depth_breadth.record_entry_return()
            else:
                goal = self._region_return_goal(
                    entry, pose, grid, "region_coverage_entry_return",
                    exclusion.get("coverage"))
                if goal is not None:
                    if self.first_goal_elapsed is None:
                        self.first_goal_elapsed = self.elapsed()
                    return goal, {
                        "source": "region_coverage_entry_return",
                        "coverage": exclusion.get("coverage"),
                        "relocated": goal.get("region_entry_relocated"),
                    }, 0
        temporary = tempfile.mkdtemp(prefix="simenv_baseline_fuel_")
        history_file = os.path.join(temporary, "planner_history.csv")
        self._write_planner_history_csv(history_file)
        command = [
            "rosrun", "simenv_exploration", "fuel_lite_planner",
            "__name:=baseline_fuel_cycle_{:03d}".format(cycle),
            "_map_file:=" + self.map_file, "_output_dir:=" + temporary,
            "_frame_id:=" + frame, "_use_pose_parameters:=true",
            "_exit_after_plan:=true", "_robot_x:={:.9f}".format(pose[0]),
            "_robot_y:={:.9f}".format(pose[1]), "_robot_z:={:.9f}".format(pose[2]),
            "_history_file:=" + history_file,
            "_execution_duplicate_radius:={:.6f}".format(self.fuel_duplicate_radius),
            "_allow_transit_revisit_fallback:={}".format(
                str(self.fuel_allow_transit_revisit_fallback).lower()),
            "_enable_room_information_gain:=false", "_exploration_mode:=GENERIC",
            "/exploration_goal:=/simenv/baseline_planner_raw_goal",
        ]
        if math.isfinite(self.fuel_minimum_selected_score):
            command.append("_minimum_selected_score:={:.6f}".format(
                self.fuel_minimum_selected_score))
        if (not in_corridor and self.depth_breadth.config.enabled and
                self.depth_breadth.anchor is not None):
            command.extend([
                "_depth_breadth_phase:=" + self.depth_breadth.phase(self.elapsed()),
                "_depth_breadth_anchor_x:={:.9f}".format(self.depth_breadth.anchor[0]),
                "_depth_breadth_anchor_y:={:.9f}".format(self.depth_breadth.anchor[1]),
            ])
        if exclusion is not None:
            command.extend([
                "_enable_region_exclusion:=true",
                "_excluded_region_x:={:.9f}".format(exclusion["center"][0]),
                "_excluded_region_y:={:.9f}".format(exclusion["center"][1]),
                "_excluded_region_radius:={:.6f}".format(exclusion["radius"]),
            ])
        try:
            completed = subprocess.run(command, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL,
                                       timeout=self.planner_timeout, check=False)
            goal = self._load_json(os.path.join(
                temporary, "current_exploration_goal.json"))
            score = self._load_json(os.path.join(
                temporary, "exploration_score.json")) or {}
            candidates = self._load_json(os.path.join(
                temporary, "candidate_viewpoints.json")) or []
            raw_frontiers = self._load_json(os.path.join(
                temporary, "frontiers.json")) or []
            side_goal = self._corridor_side_coverage_goal(
                candidates, score, pose, grid, corridor, raw_frontiers)
            if side_goal is not None:
                self.corridor_sweep_history.append({
                    "event": "CORRIDOR_SIDE_COVERAGE_DEBT_SELECTED",
                    "elapsed_sec": round(self.elapsed(), 3),
                    "original_goal": (goal or {}).get("position"),
                    "replacement_goal": side_goal.get("position"),
                    "frontier_center": side_goal.get("frontier_center"),
                    "frontier_lateral_m":
                        side_goal.get("side_frontier_lateral_m"),
                    "frontier_longitudinal_m":
                        side_goal.get("side_frontier_longitudinal_m"),
                })
                goal = side_goal
            return_code = completed.returncode
        except (OSError, subprocess.TimeoutExpired):
            goal, score, return_code = None, {}, 124
        finally:
            if self.preserve_planner_diagnostics:
                diagnostic_dir = os.path.join(
                    self.output_dir, "planner_diagnostics", "cycle_{:03d}".format(cycle))
                os.makedirs(diagnostic_dir, exist_ok=True)
                for name in ("current_exploration_goal.json", "exploration_score.json",
                             "frontier_filter_diagnostic.json", "candidate_viewpoints.json",
                             "frontiers.json"):
                    source = os.path.join(temporary, name)
                    if os.path.isfile(source):
                        shutil.copy2(source, os.path.join(diagnostic_dir, name))
            shutil.rmtree(temporary, ignore_errors=True)
        self.fuel_plan_count += 1
        if (not goal and exclusion is not None and
                exclusion.get("entry_anchor") is not None and
                exclusion.get("reason") != "coverage_complete" and
                not exclusion.get("entry_return_failed", False)):
            entry = exclusion["entry_anchor"]
            if math.hypot(entry[0] - pose[0], entry[1] - pose[1]) > self.reached_tolerance:
                goal = self._region_return_goal(
                    entry, pose, grid, "depth_breadth_entry_return")
                if goal is not None:
                    score = {
                        "fallback": "reachable_region_entry_neighbour",
                        "exclusion": exclusion,
                        "relocated": goal.get("region_entry_relocated"),
                    }
                    return_code = 0
        if goal:
            goal.setdefault("source", "fuel_frontier")
            goal["scheduler_phase"] = ("CORRIDOR_SWEEP" if in_corridor else
                                       self.depth_breadth.phase(self.elapsed()))
            goal["depth_breadth_escape"] = exclusion is not None
            if exclusion is not None:
                goal["depth_breadth_exclusion"] = exclusion
        if goal and self.first_goal_elapsed is None:
            self.first_goal_elapsed = self.elapsed()
        return goal, score, return_code

    def _region_return_breadcrumbs(self, entry, current, spacing=0.90,
                                   since=None):
        with self.lock:
            trajectory = list(self.trajectory)
        anchors = recent_reverse_trajectory_anchors(
            trajectory, (float(entry[0]), float(entry[1])),
            (float(current[0]), float(current[1])), spacing,
            since=since)
        return [[float(point[0]), float(point[1])] for point in anchors]

    def _plan_region_stall_backtrack(self, cycle):
        """Retreat over recently occupied poses before retrying a long exit.

        This is used only after the executor reports commanded motion without
        localization progress inside a physically entered region.  Following
        the robot's own recent trajectory avoids trusting a newly projected
        2-D shortcut through furniture and gives SCAN-lite a non-colliding
        start pose for the next full return plan.
        """
        if (not self.region_stall_backtrack_pending or
                not self.depth_breadth.entry_observed or
                self.region_stall_backtrack_attempts >=
                self.region_stall_backtrack_limit):
            return None
        with self.lock:
            current = self.pose
            trajectory = list(self.trajectory)
        if current is None or len(trajectory) < 2:
            self.region_stall_backtrack_pending = False
            return None
        current_point = (float(current[0]), float(current[1]))
        previous = current_point
        retained = []
        last_retained = current_point
        travelled = 0.0
        entry_started = self.depth_breadth.physical_entry_started_at
        for sample in reversed(trajectory):
            if entry_started is not None and float(sample[0]) < entry_started:
                break
            point = (float(sample[1]), float(sample[2]))
            step = math.hypot(point[0] - previous[0],
                              point[1] - previous[1])
            previous = point
            # Ignore stationary jitter and do not bridge a localization jump.
            if step < 0.01:
                continue
            if step > 0.75:
                break
            travelled += step
            if math.hypot(point[0] - last_retained[0],
                          point[1] - last_retained[1]) >= \
                    self.region_stall_backtrack_spacing:
                retained.append(point)
                last_retained = point
            if (travelled >= self.region_stall_backtrack_distance and
                    math.hypot(point[0] - current_point[0],
                               point[1] - current_point[1]) >= 0.45):
                break
        if (not retained or
                math.hypot(retained[-1][0] - current_point[0],
                           retained[-1][1] - current_point[1]) < 0.45):
            self.region_stall_backtrack_pending = False
            return None
        self.region_stall_backtrack_pending = False
        self.region_stall_backtrack_attempts += 1
        target = retained[-1]
        path = [current_point] + retained
        self.depth_breadth_history.append({
            "event": "REGION_STALL_BACKTRACK_SELECTED",
            "elapsed_sec": round(self.elapsed(), 3),
            "cycle": cycle,
            "attempt": self.region_stall_backtrack_attempts,
            "target": [target[0], target[1]],
            "trajectory_distance_m": travelled,
            "waypoint_count": len(retained),
        })
        return {
            "position": [target[0], target[1], float(current[2])],
            "information_gain": 0.0,
            "score": 0.0,
            "source": "region_stall_backtrack",
            "scheduler_phase": "REGION_STALL_BACKTRACK",
            "region_stall_backtrack": True,
            "verified_trajectory_backtrack": True,
            "_preplanned_path_result": {
                "success": True,
                "reason": "recent_trajectory_backtrack",
                "path": path,
                "execution_waypoints": retained,
            },
        }

    def _plan_corridor_start_collision_backtrack(self, cycle):
        """Leave a newly occupied corridor endpoint over the driven path.

        The path is deliberately not sent through A* or SCAN-lite again: its
        first vertex is the footprint that just failed that check.  All later
        vertices are recent poses physically occupied during the immediately
        preceding successful corridor transit, and discontinuous localization
        samples are rejected.  Once clear, ordinary forward planning and all
        normal collision checks resume on the next cycle.
        """
        if (self.corridor_start_collision_backtrack_attempts >=
                self.corridor_start_collision_backtrack_limit):
            return None
        with self.lock:
            current = self.pose
            trajectory = list(self.trajectory)
        if current is None or len(trajectory) < 2:
            return None
        current_point = (float(current[0]), float(current[1]))
        previous = current_point
        retained = []
        last_retained = current_point
        travelled = 0.0
        # About ten metres / several seconds of dense odometry are ample for
        # the bounded retreat and prevent reaching an earlier room excursion.
        for sample in reversed(trajectory[-160:]):
            point = (float(sample[1]), float(sample[2]))
            step = math.hypot(point[0] - previous[0], point[1] - previous[1])
            previous = point
            if step < 0.01:
                continue
            if step > 0.75:
                break
            travelled += step
            if math.hypot(point[0] - last_retained[0],
                          point[1] - last_retained[1]) >= \
                    self.corridor_start_collision_backtrack_spacing:
                retained.append(point)
                last_retained = point
            if (travelled >=
                    self.corridor_start_collision_backtrack_distance and
                    math.hypot(point[0] - current_point[0],
                               point[1] - current_point[1]) >= 0.45):
                break
        if (not retained or
                math.hypot(retained[-1][0] - current_point[0],
                           retained[-1][1] - current_point[1]) < 0.45):
            return None
        self.corridor_start_collision_backtrack_attempts += 1
        target = retained[-1]
        self.corridor_sweep_history.append({
            "event": "CORRIDOR_START_COLLISION_BACKTRACK_SELECTED",
            "elapsed_sec": round(self.elapsed(), 3),
            "cycle": cycle,
            "attempt": self.corridor_start_collision_backtrack_attempts,
            "target": [target[0], target[1]],
            "trajectory_distance_m": travelled,
            "waypoint_count": len(retained),
            "policy": "recent_physically_traversed_path_then_forward_replan",
        })
        return {
            "position": [target[0], target[1], float(current[2])],
            "information_gain": 0.0,
            "score": 0.0,
            "source": "corridor_start_collision_backtrack",
            "scheduler_phase": "CORRIDOR_COLLISION_BACKTRACK",
            "corridor_start_collision_backtrack": True,
            "verified_trajectory_backtrack": True,
            "_preplanned_path_result": {
                "success": True,
                "reason": "recent_corridor_trajectory_backtrack",
                "path": [current_point] + retained,
                "execution_waypoints": retained,
            },
        }

    def _room_revisit_scan_signature(self, pose):
        """Return a rotation-normalized local LiDAR range fingerprint.

        ``cloud_registered`` is map-framed, but subtracting the simultaneous
        odometry pose and rotating by yaw makes this descriptor local.  It is
        therefore insensitive to global translation drift while retaining the
        doorway/furniture silhouette that distinguishes neighbouring rooms.
        """
        if not self.room_revisit_scan_enabled or pose is None:
            return None
        with self.lock:
            cloud = np.asarray(self.latest_cloud_xy, dtype=np.float32).copy()
        if len(cloud) < self.room_revisit_scan_min_bins:
            return None
        dx = cloud[:, 0] - float(pose[0])
        dy = cloud[:, 1] - float(pose[1])
        ranges = np.hypot(dx, dy)
        valid = np.isfinite(ranges) & (ranges >= 0.35) & (ranges <= 6.5)
        if int(np.count_nonzero(valid)) < self.room_revisit_scan_min_bins:
            return None
        angles = np.arctan2(dy[valid], dx[valid]) - float(pose[3])
        angles = (angles + math.pi) % (2.0 * math.pi) - math.pi
        bins = np.floor((angles + math.pi) / (2.0 * math.pi) *
                        self.room_revisit_scan_bins).astype(np.int32)
        bins = np.clip(bins, 0, self.room_revisit_scan_bins - 1)
        values = np.full(self.room_revisit_scan_bins, np.nan, dtype=np.float32)
        for index, distance in zip(bins, ranges[valid]):
            if not np.isfinite(values[index]) or distance < values[index]:
                values[index] = distance
        return values

    def _matching_completed_room_scan(self, signature, door):
        """Match only same-facing completed portals using local scan shape."""
        if signature is None or door is None:
            return None
        for known in self.room_scheduler.detector.doors:
            if not (known.completed or known.visited):
                continue
            record = self.room_revisit_scan_records.get(known.door_id)
            if record is None:
                continue
            if (abs(float(door.width) - float(record["width"])) >
                    self.room_revisit_scan_max_width_delta):
                continue
            alignment = math.cos(float(door.normal_direction) -
                                 float(record["normal_direction"]))
            if alignment < 0.85:
                continue
            reference = np.asarray(record["signature"], dtype=np.float32)
            overlap = np.isfinite(signature) & np.isfinite(reference)
            count = int(np.count_nonzero(overlap))
            if count < self.room_revisit_scan_min_bins:
                continue
            error = float(np.median(np.abs(signature[overlap] -
                                           reference[overlap])))
            if error <= self.room_revisit_scan_max_median_error:
                return {"door_id": known.door_id, "overlap_bins": count,
                        "median_range_error_m": error,
                        "normal_alignment": alignment}
        return None

    def _record_room_scan_signature(self, door, signature):
        if door is None or signature is None:
            return
        self.room_revisit_scan_records[str(door.door_id)] = {
            "signature": signature.tolist(), "width": float(door.width),
            "normal_direction": float(door.normal_direction),
            "elapsed_sec": round(self.elapsed(), 3),
        }

    def _log_room_revisit_guard(self, payload):
        try:
            with open(self.room_revisit_scan_log_path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, sort_keys=True) + "\n")
        except OSError:
            pass

    def _plan_room_goal(self):
        """Recognize a doorway and emit ENTRY, centre, left, and right goals."""
        if not self.room_scheduler.config.enabled:
            return None
        with self.lock:
            pose, grid = self.pose, self.grid
            trajectory = [(sample[1], sample[2]) for sample in self.trajectory]
        if pose is None or grid is None:
            return None
        if (self.room_scheduler.active_door is None and
                not self.corridor_door_detection_armed):
            return None
        detected = self.room_scheduler.observe(
            grid, trajectory, self.elapsed(),
            corridor_axis=self.corridor_station_axis,
            maximum_corridor_alignment=
                self.room_door_maximum_corridor_alignment)
        if detected is None:
            detected = self._consider_active_semantic_side_door(
                grid, pose)
        if detected is not None:
            signature = self._room_revisit_scan_signature(pose)
            duplicate = self._matching_completed_room_scan(signature, detected)
            guard_event = {
                "elapsed_sec": round(self.elapsed(), 3),
                "candidate_door_id": detected.door_id,
                "candidate_center": [float(detected.center[0]),
                                     float(detected.center[1])],
                "candidate_width_m": float(detected.width),
                "signature_available": signature is not None,
            }
            if duplicate is not None:
                guard_event.update({"event": "ROOM_REVISIT_REJECTED",
                                    "matched": duplicate})
                self._log_room_revisit_guard(guard_event)
                # Activation has not crossed the new portal yet, so aborting
                # here cannot increase the completed-room count.  It applies
                # a local cooldown and releases the forward corridor planner.
                self.room_scheduler.abort_active_room(
                    self.elapsed(), "local_lidar_room_revisit_match")
                self.corridor_sweep_history.append(dict(guard_event))
                rospy.logwarn("Rejected drifted duplicate doorway %s; matched %s "
                              "(local scan median error %.2f m)",
                              detected.door_id, duplicate["door_id"],
                              duplicate["median_range_error_m"])
                return None
            guard_event["event"] = "ROOM_REVISIT_REFERENCE_RECORDED"
            self._log_room_revisit_guard(guard_event)
            self._record_room_scan_signature(detected, signature)
            rospy.loginfo(
                "ROOM_ENTERED via %s center=(%.2f, %.2f) width=%.2f confidence=%.2f",
                detected.door_id, detected.center[0], detected.center[1],
                detected.width, detected.confidence)
        goal = self.room_scheduler.next_goal(
            grid, (pose[0], pose[1]), self.elapsed())
        if goal is not None:
            goal["position"][2] = pose[2]
            goal["scheduler_phase"] = "ROOM_" + str(goal.get("room_role", "UNKNOWN"))
            role = str(goal.get("room_role", ""))
            mandatory = goal.get("mandatory_portal_waypoints") or []
            if role == "RETURN":
                target = goal.get("position") or []
                preplanned = goal.get("_preplanned_path_result") or {}
                return_preflight_verified = bool(
                    goal.get("return_preflight_verified") and
                    preplanned.get("path"))
                anchors = ([] if return_preflight_verified else
                           (self._region_return_breadcrumbs(
                               target, pose, spacing=0.60,
                               since=self.room_scheduler.room_started_at)
                            if len(target) >= 2 else []))
                if anchors and not return_preflight_verified:
                    goal["_preplanned_path_result"] = {
                        "success": True,
                        "reason": "reverse_actual_room_trajectory_to_portal",
                        "path": [(float(pose[0]), float(pose[1]))] +
                                [tuple(point) for point in anchors],
                        "execution_waypoints":
                            [tuple(point) for point in anchors],
                    }
                    goal["room_trajectory_backtrack"] = True
                    # Every breadcrumb is a recent FAST-LIO pose physically
                    # occupied during this same room visit.  Re-running the
                    # newly inflated 2-D map over that reverse segment can
                    # reject it after furniture is observed, trapping the
                    # robot in the room.  Treat only this return-to-inside-
                    # anchor segment as verified.  The subsequent doorway
                    # EXIT remains a separate SCAN-lite checked path.
                    goal["verified_trajectory_backtrack"] = True
                timeout_path = list(
                    (goal.get("_preplanned_path_result") or {}).get(
                        "path") or [])
                if not timeout_path and len(target) >= 2:
                    timeout_path = [
                        (float(pose[0]), float(pose[1])),
                        (float(target[0]), float(target[1])),
                    ]
                return_length = polyline_length(timeout_path)
                return_timeout = exit_timeout_for_path(
                    return_length, self.room_scheduler.config)
                goal["execution_timeout_sec"] = return_timeout
                goal["progress_timeout_sec"] = max(
                    float(goal.get("progress_timeout_sec", 0.0)),
                    self.room_scheduler.config.exit_progress_timeout_seconds)
                goal["return_remaining_path_length_m"] = return_length
                goal["return_dynamic_timeout_sec"] = return_timeout
                self.room_scheduler.events.append({
                    "event": "ROOM_RETURN_PATH_PREPARED",
                    "elapsed_sec": round(self.elapsed(), 3),
                    "room_id": goal.get("room_id"),
                    "door_id": goal.get("estimated_door_id"),
                    "attempt": goal.get("room_return_attempt"),
                    "remaining_path_length_m": round(return_length, 3),
                    "execution_timeout_sec": round(return_timeout, 3),
                    "astar_preflight_verified":
                        return_preflight_verified,
                    "breadcrumb_waypoint_count": len(anchors),
                })
            elif role == "EXIT" and len(mandatory) >= 2:
                # RETURN normally leaves the robot on the inside door-normal
                # centreline. Do not replace that short verified portal path
                # with historical breadcrumbs. On a retry farther inside,
                # reverse only the current room visit and stop at the latest
                # visit to the inside anchor; this prevents 12 -> 16 -> 18
                # waypoint growth across repeated EXIT attempts.
                inside = mandatory[0]
                distance_to_inside = math.hypot(
                    float(pose[0]) - float(inside[0]),
                    float(pose[1]) - float(inside[1]))
                anchor_skipped = bool(
                    distance_to_inside <=
                    self.room_scheduler.config.exit_anchor_skip_distance)
                if anchor_skipped:
                    # Already between the deep alignment anchor and portal:
                    # replaying the old inside anchor first sends the robot
                    # back into the room. Keep only centre -> corridor.
                    mandatory = list(mandatory[1:])
                    goal["mandatory_portal_waypoints"] = mandatory
                    goal["_preplanned_path_result"] = {
                        "success": True,
                        "reason": "near_portal_skip_deeper_inside_anchor",
                        "path": [(float(pose[0]), float(pose[1]))] +
                                [(float(point[0]), float(point[1]))
                                 for point in mandatory],
                        "execution_waypoints": [
                            (float(point[0]), float(point[1]))
                            for point in mandatory],
                    }
                portal_preflight_verified = bool(
                    goal.get("portal_preflight_verified"))
                prepared_result = goal.get("_preplanned_path_result") or {}
                prepared_reason = str(prepared_result.get("reason", ""))
                # Only this preflight includes a collision-free route from
                # the deep-room pose; doorway traces need breadcrumbs first.
                current_to_portal_preflight = bool(
                    portal_preflight_verified and
                    prepared_reason == "centered_exit_preflight_path")
                # A normal preflight path is already safe.  A reverse-entry
                # fallback, however, only contains the short doorway trace;
                # it does not contain the obstacle-avoiding route from the
                # current deep-room viewpoint back to that trace.  Reuse the
                # actual room trajectory for precisely that prefix.
                verified_exit_backtrack = bool(
                    goal.get("verified_trajectory_backtrack"))
                anchors = ([] if (anchor_skipped or
                                  current_to_portal_preflight) else
                           self._region_return_breadcrumbs(
                               inside, pose, spacing=0.45,
                               since=self.room_scheduler.room_started_at))
                if (anchors and not anchor_skipped and
                        (not current_to_portal_preflight or
                         verified_exit_backtrack)):
                    path = [(float(pose[0]), float(pose[1]))]
                    path.extend(tuple(point) for point in anchors)
                    if math.hypot(path[-1][0] - float(inside[0]),
                                  path[-1][1] - float(inside[1])) > 0.08:
                        path.append((float(inside[0]), float(inside[1])))
                    path.extend((float(point[0]), float(point[1]))
                                for point in mandatory[1:])
                    goal["_preplanned_path_result"] = {
                        "success": True,
                        "reason":
                            "recent_room_trajectory_then_centered_portal",
                        "path": path,
                    }
                    goal["room_trajectory_backtrack"] = True
                timeout_path = list(
                    (goal.get("_preplanned_path_result") or {}).get(
                        "path") or [])
                if not timeout_path:
                    timeout_path = [(float(pose[0]), float(pose[1]))]
                    timeout_path.extend((float(point[0]), float(point[1]))
                                        for point in mandatory)
                remaining_length = polyline_length(timeout_path)
                dynamic_timeout = exit_timeout_for_path(
                    remaining_length, self.room_scheduler.config)
                goal["execution_timeout_sec"] = dynamic_timeout
                goal["exit_remaining_path_length_m"] = remaining_length
                goal["exit_dynamic_timeout_sec"] = dynamic_timeout
                goal["exit_anchor_skipped"] = anchor_skipped
                self.room_scheduler.events.append({
                    "event": "ROOM_EXIT_PATH_PREPARED",
                    "elapsed_sec": round(self.elapsed(), 3),
                    "room_id": goal.get("room_id"),
                    "door_id": goal.get("estimated_door_id"),
                    "attempt": goal.get("room_exit_attempt"),
                    "remaining_path_length_m": round(remaining_length, 3),
                    "execution_timeout_sec": round(dynamic_timeout, 3),
                    "distance_to_inside_anchor_m": round(
                        distance_to_inside, 3),
                    "inside_anchor_skipped": anchor_skipped,
                    "breadcrumb_waypoint_count": len(anchors),
                })
            rospy.loginfo("Room %s selected %s at (%.2f, %.2f)",
                          goal.get("room_id"), goal.get("room_role"),
                          goal["position"][0], goal["position"][1])
        return goal

    def _consider_active_semantic_side_door(self, grid, pose):
        """Promote one deeply observed side branch before returning corridor.

        The branch scheduler supplies the semantic evidence requested by the
        exploration policy: stable corridor station, side frontier, repeated
        observations and a physically executed A*/SCAN-safe penetration.  A
        separate observed-jamb candidate and normal portal preflight remain
        mandatory, so this cannot manufacture a virtual room coordinate.
        """
        if (not self.corridor_door_detection_armed or
                not self.enable_semantic_side_door_candidate or
                self.room_scheduler.active_door is not None or
                self.corridor_station_axis is None):
            return None
        branch = self.branch_scheduler.find(
            self.branch_scheduler.active_branch_id)
        if (branch is None or
                branch.state != "ENTERED_PARTIAL" or
                branch.observations <
                self.semantic_side_door_minimum_observations or
                branch.entry_quality <
                self.semantic_side_door_minimum_penetration):
            return None
        axis = np.asarray(self.corridor_station_axis, dtype=float)
        axis /= max(1e-9, float(np.linalg.norm(axis)))
        normal = np.asarray([-axis[1], axis[0]], dtype=float)
        centerline = (
            np.asarray(branch.entry_target, dtype=float) -
            float(branch.side) * float(branch.entry_quality) * normal)
        signed_depth = float(np.dot(
            np.asarray([float(pose[0]), float(pose[1])]) - centerline,
            float(branch.side) * normal))
        if signed_depth < self.semantic_side_door_minimum_penetration - 0.25:
            return None
        with self.lock:
            observed_update = int(self.grid_update_count)
        event_start = len(self.room_scheduler.events)
        confirmed = None
        for confirmation_index in range(
                self.room_scheduler.config.candidate_confirmation_count):
            if confirmation_index:
                deadline = time.monotonic() + 2.0
                while not rospy.is_shutdown() and time.monotonic() < deadline:
                    with self.lock:
                        if self.grid_update_count > observed_update:
                            grid = self.grid
                            pose = self.pose
                            observed_update = self.grid_update_count
                            break
                    time.sleep(0.03)
                else:
                    break
            confirmed = self.room_scheduler.consider_semantic_corridor_branch(
                grid, (float(pose[0]), float(pose[1])),
                (float(centerline[0]), float(centerline[1])), axis,
                branch.side, self.elapsed(), branch.branch_id)
            if confirmed is not None:
                break
        if confirmed is None:
            new_events = self.room_scheduler.events[event_start:]
            self.corridor_sweep_history.append({
                "event": "SEMANTIC_SIDE_DOOR_EVALUATED",
                "elapsed_sec": round(self.elapsed(), 3),
                "branch_id": branch.branch_id,
                "side": branch.side,
                "observations": branch.observations,
                "entry_quality_m": branch.entry_quality,
                "current_penetration_m": signed_depth,
                "confirmed": False,
                "door_events": [item.get("event") for item in new_events],
                "reasons": [item.get("reason") for item in new_events
                            if item.get("reason")],
            })
            return None
        self.corridor_sweep_history.append({
            "event": "SEMANTIC_SIDE_DOOR_ACTIVATED",
            "elapsed_sec": round(self.elapsed(), 3),
            "branch_id": branch.branch_id,
            "side": branch.side,
            "observations": branch.observations,
            "entry_quality_m": branch.entry_quality,
            "door_id": confirmed.door_id,
            "door_center": list(confirmed.center),
        })
        return confirmed

    def _mission_rooms_complete(self):
        return (self.room_scheduler.active_door is None and
                sum(door.completed for door in
                    self.room_scheduler.detector.doors) >= self.room_target_count)

    def _mission_rooms_physically_exited(self):
        """Return true once the required number of ENTRY/EXIT crossings exists.

        Visual/LiDAR coverage quality remains useful while selecting room
        viewpoints, but it must not block the first-floor-to-stair handoff
        after every required room has a confirmed physical EXIT.
        """
        return (self.room_scheduler.active_door is None and
                sum(bool(door.visited) for door in
                    self.room_scheduler.detector.doors) >= self.room_target_count)

    def _terminal_return_room_supplement_active(self):
        """Whether the homeward corridor pass must still recover rooms.

        Reaching the outbound terminal wall only changes the search direction;
        it is not floor completion. During this pass, fresh side-door evidence
        is intentionally allowed behind the outbound high-water station.
        """
        return bool(
            self.corridor_terminal_return_latched and
            getattr(self, "corridor_partial_return_trigger_reason", None) !=
            "next_floor_time_reserve" and
            self.room_scheduler.active_door is None and
            not self._mission_rooms_physically_exited())

    def _known_unvisited_room_doors(self):
        """Return registered portals that have never completed a real EXIT."""
        return [door for door in self.room_scheduler.detector.doors
                if not bool(door.visited) and not bool(door.completed)]

    def _terminal_return_has_blocking_known_door(self, all_rooms_exited):
        return bool(
            getattr(self,
                    "terminal_return_blocked_by_known_unvisited_door", False) and
            not all_rooms_exited and self._known_unvisited_room_doors())

    def _plan_known_unvisited_door_retry_goal(self):
        """Retry a nearby failed portal before any new corridor sweep."""
        if (not getattr(self, "enable_known_unvisited_door_retry", False) or
                (getattr(
                    self,
                    "known_unvisited_door_retry_only_on_terminal_return",
                    False) and not self.corridor_terminal_return_latched) or
                not self.corridor_door_detection_armed or
                self.room_scheduler.active_door is not None):
            return None
        with self.lock:
            pose, grid = self.pose, self.grid
        if (pose is None or grid is None or
                not self._on_station_corridor_centerline(pose)):
            return None
        door = self.room_scheduler.retry_known_unvisited_door(
            grid, (float(pose[0]), float(pose[1])), self.elapsed(),
            self.known_unvisited_door_retry_distance)
        if door is None:
            return None
        goal = self.room_scheduler.next_goal(
            grid, (float(pose[0]), float(pose[1])), self.elapsed())
        if goal is None:
            return None
        goal["position"][2] = float(pose[2])
        goal["scheduler_phase"] = "ROOM_ENTRY"
        goal["door_takeover_source"] = "known_unvisited_door_retry"
        self.corridor_sweep_history.append({
            "event": "KNOWN_UNVISITED_DOOR_RETRY_ACTIVATED",
            "elapsed_sec": round(self.elapsed(), 3),
            "door_id": door.door_id,
            "door_center": [float(door.center[0]), float(door.center[1])],
            "goal": list(goal.get("position", [])),
            "policy": "retry_before_corridor_sweep",
        })
        return goal

    def _stair_return_handoff_authorized(self):
        """Whether the current return may transfer ownership to the stairs.

        A terminal wall starts a best-effort reverse room-search pass. Fresh
        doors may preempt that return, but missed rooms cannot permanently
        block the multi-floor mission once the stair gate is reached.
        """
        exited = sum(bool(door.visited) for door in
                     self.room_scheduler.detector.doors)
        return bool(
            self.stair_handoff_on_corridor_exit and
            self.room_scheduler.active_door is None and
            self._floor_handoff_room_requirement_met() and
            (self._mission_rooms_physically_exited() or
             self.corridor_terminal_return_latched or exited >= 1))

    def _stair_return_deadline(self):
        """Deadline including grace only after stair return is authorized."""
        grace_active = bool(
            self.stair_return_transit_announced and
            self._stair_return_handoff_authorized())
        return (self.maximum_duration +
                (self.stair_return_grace_seconds if grace_active else 0.0))

    def _stair_return_remaining(self):
        return max(0.0, self._stair_return_deadline() - self.elapsed())

    def _announce_stair_return_transit(self, reason):
        if not self._floor_handoff_room_requirement_met():
            # 拒绝但允许重试: 探索可能继续补全出口, 4/4 满足后必须能触发。
            # 原一次性 strict_stair_transit_rejection_logged 会永久封死交接:
            # 3/4 被拒一次后即使探索补全到 4/4 也不再尝试, 死锁到 TIME_LIMIT
            # (批次16 RUN2/3 STAIR_HANDOFF_NOT_REACHED 根因之一)。
            # 5s 节流仅限日志, 不影响每次重新评估。
            if (time.monotonic() -
                    getattr(self, "_stair_transit_reject_logged_at", 0.0)
                    >= 5.0):
                self._stair_transit_reject_logged_at = time.monotonic()
                exited_rooms = sum(
                    bool(door.visited)
                    for door in self.room_scheduler.detector.doors)
                self.corridor_sweep_history.append({
                    "event": "STAIR_RETURN_TRANSIT_REJECTED_MISSING_ROOMS",
                    "elapsed_sec": round(self.elapsed(), 3),
                    "exited_rooms": exited_rooms,
                    "required_rooms": max(4, int(self.room_target_count)),
                    "requested_reason": reason,
                })
                rospy.logwarn(
                    "Strict floor handoff rejected stair return transit "
                    "(retry allowed): physical exits %d/%d", exited_rooms,
                    max(4, int(self.room_target_count)))
            return False
        """Arm the physical stair-side lobby gate exactly once per mission."""
        if self.stair_return_transit_announced:
            return
        self.stair_return_transit_announced = True
        self.stair_return_transit_pub.publish(Bool(data=True))
        self._set_state("STAIR_RETURN_TRANSIT", reason)

    def _recent_room_exit_side(self, door_center, pose, corridor):
        """Return the stable corridor side of a recently exited room."""
        if (self.last_room_exit_side is None or
                self.elapsed() > self.room_exit_resume_until or
                self.last_room_exit_pose is None or pose is None or
                corridor is None):
            return None
        distance = math.hypot(float(pose[0]) - self.last_room_exit_pose[0],
                              float(pose[1]) - self.last_room_exit_pose[1])
        if distance >= self.room_exit_resume_distance:
            return None
        return int(self.last_room_exit_side)

    def _recent_room_exit_opposite_side(self, pose):
        """Return a short-range exit-side reference for an opposite portal.

        Same-side re-entry protection intentionally expires after 1.2 m.
        A real opposite doorway can appear only while that clearance motion
        is under way, so its evidence gets a separate, bounded recenter
        window; it never authorizes a generic rearward doorway.
        """
        if (self.last_room_exit_side is None or
                self.last_room_exit_pose is None or pose is None or
                self.elapsed() > (self.room_exit_resume_until +
                                  self.room_exit_opposite_door_revisit_time)):
            return None
        distance = math.hypot(float(pose[0]) - self.last_room_exit_pose[0],
                              float(pose[1]) - self.last_room_exit_pose[1])
        if distance > self.room_exit_opposite_door_revisit_radius:
            return None
        return int(self.last_room_exit_side)

    def _plan_local_room_entry_goal(self):
        """Select an independently detected, locally verified side opening.

        This consumes candidates but never infers a room from a global y value
        or a stored longitudinal building segment. The normal A*/SCAN path
        pipeline validates the returned known-free endpoint again.
        """
        if (not self.corridor_door_detection_armed or
                not self.enable_local_doorway_detector or
                self.room_scheduler.active_door is not None):
            return None
        with self.lock:
            status, pose, grid = self.local_entry_status, self.pose, self.grid
            received = self.last_local_entry_received
        if (status is None or pose is None or
                self.elapsed() - received > self.local_candidate_freshness):
            return None
        corridor = self._door_search_corridor_context(
            pose, grid, "local_room_entry")
        if corridor is None:
            return None
        if not self._on_station_corridor_centerline(pose):
            return None
        if not self._at_corridor_forward_high_water(pose, corridor):
            return None
        corridor_axis, corridor_normal = self._stable_corridor_axes(
            corridor["axis"])
        corridor_center = np.asarray(
            corridor["centerline_point"], dtype=float)
        recent_exit_side = self._recent_room_exit_side(
            None, pose, corridor)
        recent_exit_opposite_side = self._recent_room_exit_opposite_side(pose)
        ranked = []
        for candidate in status.get("candidates") or []:
            point = candidate.get("entry_goal") or []
            if (len(point) < 2 or not candidate.get("confirmed") or
                    not candidate.get("astar_reachable") or
                    not candidate.get("scan_lite_safe") or
                    float(candidate.get("width_m", math.inf)) >
                    self.local_door_maximum_semantic_width or
                    not self._local_door_has_room_area(candidate)):
                continue
            if self.room_scheduler.goal_in_visited_room(
                    (float(point[0]), float(point[1]))):
                continue
            if self._visited_component_for_corridor_goal(
                    point, pose, grid).get("matched"):
                continue
            try:
                candidate_normal = np.asarray([
                    math.cos(float(candidate["yaw"])),
                    math.sin(float(candidate["yaw"]))], dtype=float)
                door_center = np.asarray(candidate["door_center"], dtype=float)
            except (KeyError, TypeError, ValueError):
                continue
            relative = door_center - corridor_center
            lateral = abs(float(np.dot(relative, corridor_normal)))
            longitudinal = abs(float(np.dot(relative, corridor_axis)))
            stable_side = (1 if float(np.dot(relative, corridor_normal)) >= 0.0
                           else -1)
            if (recent_exit_side is not None and
                    stable_side == recent_exit_side):
                # The just-exited side gets a short forward-clearance window;
                # this is not a global blacklist and does not affect the
                # opposite side.
                continue
            door_station = self._corridor_station(door_center)
            if self._reject_initial_lobby_throat(
                    door_center, candidate.get("candidate_id"),
                    "local_room_entry"):
                continue
            if not self._door_centerline_lateral_admissible(
                    door_center, "local_room_entry",
                    candidate.get("candidate_id")):
                continue
            rearward, directed_delta = self._new_door_is_behind_corridor_progress(
                door_center, pose)
            paired_opposite = self._is_confirmed_paired_opposite_side(
                door_station, stable_side)
            # A room directly opposite the just-exited portal can be visible
            # only after clearing that portal.  Admit it only with strong
            # local temporal evidence; weak one/two-frame wall-return tracks
            # must not steal the forward corridor objective.
            recent_exit_opposite = bool(
                rearward and recent_exit_opposite_side is not None and
                stable_side != recent_exit_opposite_side and
                directed_delta >= -self.room_exit_opposite_door_revisit_radius and
                self.last_room_exit_pose is not None and
                # The local detector and FAST-LIO can disagree by roughly
                # two metres along the corridor while clearing a doorway.
                # This wider tolerance is only for the bounded recent-exit
                # opposite-side recenter, never for normal rearward doors.
                abs(door_station - self._corridor_station(
                    self.last_room_exit_pose)) <=
                self.room_exit_opposite_door_revisit_radius and
                bool(candidate.get("confirmed")))
            # Once one side of a corridor station has been visited, the
            # unvisited opposite-side door remains a valid next task even if
            # the robot has rolled slightly past its longitudinal station
            # while clearing the first exit.  Do not apply this exception to
            # arbitrary rearward openings: it requires the branch scheduler's
            # observed same-station/opposite-side evidence.
            # ``recent_exit_opposite`` is deliberately stricter than the
            # normal paired-branch test (same station, other side, >=6
            # observations, and a bounded 2 m rollback).  It must therefore
            # participate in the admission decision as well.  Previously it
            # was only used to choose a log label, so a valid opposite door
            # was still rejected by the generic "behind progress" guard.
            allow_paired_rearward = bool(
                recent_exit_opposite or
                (rearward and
                 directed_delta >= -self.paired_opposite_maximum_backtrack and
                 paired_opposite and
                 self._rearward_branch_is_paired_opposite(
                     door_station, stable_side,
                     self._corridor_station(pose))))
            # During a latched terminal-to-G2 return, a locally confirmed,
            # unvisited side opening is precisely the allowed recovery work.
            # It is still constrained by the same A*/SCAN-lite, width and
            # room-area checks below; this never permits a generic rearward
            # frontier or lobby goal.
            terminal_return_recovery = bool(
                self._terminal_return_room_supplement_active() or
                self._missing_room_retrace_active())
            if rearward and not (allow_paired_rearward or
                                 terminal_return_recovery):
                self.corridor_sweep_history.append({
                    "event": "LOCAL_ROOM_ENTRY_REJECTED_BEHIND_PROGRESS",
                    "elapsed_sec": round(self.elapsed(), 3),
                    "candidate_id": candidate.get("candidate_id"),
                    "door_center": list(door_center),
                    "directed_delta_m": round(directed_delta, 3),
                    "maximum_lookbehind_m": self.room_new_door_maximum_lookbehind,
                })
                continue
            if allow_paired_rearward:
                self.corridor_sweep_history.append({
                    "event": ("LOCAL_ROOM_ENTRY_RECENT_EXIT_OPPOSITE_ALLOWED"
                              if recent_exit_opposite else
                              "LOCAL_ROOM_ENTRY_PAIRED_OPPOSITE_ALLOWED"),
                    "elapsed_sec": round(self.elapsed(), 3),
                    "candidate_id": candidate.get("candidate_id"),
                    "door_center": list(door_center),
                    "directed_delta_m": round(directed_delta, 3),
                })
            # Local candidates are actionable doors only while observed from
            # the established corridor and with an approximately lateral
            # inward normal. This prevents furniture gaps inside a room from
            # being registered as another doorway.
            if (abs(float(np.dot(candidate_normal, corridor_axis))) >
                    self.local_door_maximum_corridor_alignment or
                    not (0.45 <= lateral <= 3.20) or longitudinal > 3.50):
                continue
            identifier = str(candidate.get("candidate_id", "local-door"))
            previous = self.local_candidate_attempts.get(identifier, -math.inf)
            if self.elapsed() - previous < 45.0:
                continue
            distance = math.hypot(float(point[0]) - pose[0],
                                  float(point[1]) - pose[1])
            score = (float(candidate.get("unknown_behind_m2", 0.0)) +
                     0.35 * float(candidate.get("free_behind_m2", 0.0)) -
                     0.20 * distance +
                     (4.0 if (paired_opposite or recent_exit_opposite) else 0.0))
            if (recent_exit_side is not None and
                    stable_side != recent_exit_side):
                score += 6.0
            ranked.append((score, candidate, distance,
                           paired_opposite or recent_exit_opposite))
        if not ranked:
            return None
        for _, candidate, distance, paired_opposite in sorted(
                ranked, key=lambda item: item[0], reverse=True):
            identifier = str(candidate.get("candidate_id", "local-door"))
            self.local_candidate_attempts[identifier] = self.elapsed()
            point = candidate["entry_goal"]
            activated = None
            if self.room_scheduler.config.enabled:
                activated = self.room_scheduler.consider_local_door_candidate(
                    grid, (float(pose[0]), float(pose[1])),
                    candidate, self.elapsed())
                if activated is None:
                    # Common portal validation may be temporarily blocked by
                    # a still-growing map. Match its short retry debounce,
                    # rather than blacklisting a real local aperture for 45s.
                    self.local_candidate_attempts[identifier] = (
                        self.elapsed() - 40.0)
                    continue
                goal = self.room_scheduler.next_goal(
                    grid, (float(pose[0]), float(pose[1])),
                    self.elapsed())
                if goal is None:
                    continue
                goal["position"][2] = float(pose[2])
                goal["scheduler_phase"] = "LOCAL_DOOR_COMMIT"
                goal["door_takeover_source"] = "local_doorway_detector"
                goal["local_door_candidate_id"] = identifier
                goal["local_door_evidence"] = candidate
            else:
                goal = {
                    "position": [float(point[0]), float(point[1]),
                                 float(pose[2])],
                    "information_gain": float(
                        candidate.get("unknown_behind_m2", 0.0)),
                    "score": float(candidate.get("open_area_m2", 0.0)),
                    "source": "local_doorway_detector",
                    "scheduler_phase": "LOCAL_ROOM_ENTRY",
                    "local_door_candidate_id": identifier,
                    "local_door_evidence": candidate,
                }
            self.corridor_sweep_history.append({
                "event": "LOCAL_ROOM_ENTRY_CANDIDATE_SELECTED",
                "elapsed_sec": round(self.elapsed(), 3),
                "candidate_id": identifier,
                "entry_goal": list(point),
                "door_center": candidate.get("door_center"),
                "side": candidate.get("side"),
                "unknown_behind_m2": candidate.get("unknown_behind_m2"),
                "distance_m": distance,
                "paired_opposite_door_priority": paired_opposite,
                "evidence_frame": "robot_local_8m_geometry",
                "room_scheduler_activated": activated is not None,
            })
            return goal
        return None

    def _plan_pending_paired_opposite_door(self):
        """Recenter once for a confirmed, nearby opposite-side doorway."""
        pending = self.pending_paired_opposite_door
        if not isinstance(pending, dict):
            return None
        if self.room_scheduler.active_door is not None:
            self.pending_paired_opposite_door = None
            return None
        with self.lock:
            pose, grid = self.pose, self.grid
        target = pending.get("corridor_side") or []
        if (pose is None or grid is None or len(target) < 2 or
                self.elapsed() - float(pending.get(
                    "stored_elapsed", -math.inf)) >
                self.paired_opposite_recenter_pending_timeout):
            self.pending_paired_opposite_door = None
            return None
        distance = math.hypot(float(target[0]) - pose[0],
                              float(target[1]) - pose[1])
        if distance > max(self.reached_tolerance, 0.30):
            self.corridor_sweep_history.append({
                "event": "PAIRED_OPPOSITE_DOOR_RECENTER_SELECTED",
                "elapsed_sec": round(self.elapsed(), 3),
                "candidate_id": pending.get("candidate_id"),
                "target": [float(target[0]), float(target[1])],
                "distance_m": round(distance, 3),
            })
            return {
                "position": [float(target[0]), float(target[1]), float(pose[2])],
                "information_gain": 0.0,
                "score": 0.0,
                "source": "paired_opposite_door_recenter",
                "scheduler_phase": "PAIRED_OPPOSITE_DOOR_RECENTER",
                "paired_opposite_door_evidence": dict(pending),
            }
        activated = self.room_scheduler.consider_local_door_candidate(
            grid, (float(pose[0]), float(pose[1])), pending, self.elapsed())
        if activated is None:
            identifier = str(pending.get("candidate_id", "local-door"))
            self.local_candidate_attempts[identifier] = self.elapsed()
            self.corridor_sweep_history.append({
                "event": "PAIRED_OPPOSITE_DOOR_RECENTER_ENTRY_REJECTED",
                "elapsed_sec": round(self.elapsed(), 3),
                "candidate_id": identifier,
                "cooldown_seconds": 20.0,
            })
            self.pending_paired_opposite_door = None
            return None
        goal = self.room_scheduler.next_goal(
            grid, (float(pose[0]), float(pose[1])), self.elapsed())
        self.pending_paired_opposite_door = None
        if goal is not None:
            goal["position"][2] = float(pose[2])
            goal["scheduler_phase"] = "PAIRED_OPPOSITE_DOOR_ENTRY"
            goal["door_takeover_source"] = "paired_opposite_door_recenter"
        return goal

    def _local_door_preemption_candidate(self):
        """Return one stable corridor-side opening seen during goal execution."""
        if (not self.enable_local_doorway_detector or
                self.room_scheduler.active_door is not None):
            return None
        with self.lock:
            status, pose, grid = self.local_entry_status, self.pose, self.grid
            received = self.last_local_entry_received
        if (status is None or pose is None or grid is None or
                self.elapsed() - received > self.local_candidate_freshness):
            return None
        if not self.corridor_door_detection_armed:
            # A long corridor waypoint can otherwise keep the planner loop out
            # of _corridor_context() until the goal ends. Refresh the existing
            # corridor/door phase here so a confirmed door can safely preempt
            # that waypoint. All original progress, entry-lock, geometry and
            # path-preflight gates remain authoritative.
            phase_context = self._corridor_context(pose, grid)
            if not self.corridor_door_detection_armed:
                self._update_corridor_door_phase(
                    pose, phase_context or {
                        "axis": [1.0, 0.0],
                        "raw_is_corridor": False,
                        "centerline_point": [float(pose[0]), float(pose[1])],
                    })
            if not self.corridor_door_detection_armed:
                return None
            self.corridor_sweep_history.append({
                "event": "CORRIDOR_DOOR_PHASE_ARMED_DURING_ACTIVE_GOAL",
                "elapsed_sec": round(self.elapsed(), 3),
                "policy": "existing_online_corridor_and_strict_door_gates",
            })
        corridor = self._door_search_corridor_context(
            pose, grid, "local_door_preemption")
        if corridor is None:
            return None
        if not self._on_station_corridor_centerline(pose):
            return None
        if not self._at_corridor_forward_high_water(pose, corridor):
            return None
        axis, normal = self._stable_corridor_axes(corridor["axis"])
        centerline = np.asarray(corridor["centerline_point"], dtype=float)
        recent_exit_side = self._recent_room_exit_side(None, pose, corridor)
        recent_exit_opposite_side = self._recent_room_exit_opposite_side(pose)
        for candidate in status.get("candidates") or []:
            # The detector can retain a corridor-side point which has become
            # occupied after map inflation.  Give a failed recenter point a
            # short cooldown so it cannot repeatedly cancel forward progress.
            identifier = str(candidate.get("candidate_id", "local-door"))
            previous = self.local_candidate_attempts.get(identifier, -math.inf)
            if self.elapsed() - previous < 20.0:
                continue
            point = candidate.get("entry_goal") or []
            try:
                inward = np.asarray([
                    math.cos(float(candidate["yaw"])),
                    math.sin(float(candidate["yaw"]))], dtype=float)
                center = np.asarray(candidate["door_center"], dtype=float)
            except (KeyError, TypeError, ValueError):
                continue
            relative = center - centerline
            stable_side = (1 if float(np.dot(relative, normal)) >= 0.0
                           else -1)
            door_station = self._corridor_station(center)
            if self._reject_initial_lobby_throat(
                    center, candidate.get("candidate_id"),
                    "local_door_preemption"):
                continue
            if not self._door_centerline_lateral_admissible(
                    center, "local_door_preemption",
                    candidate.get("candidate_id")):
                continue
            rearward, directed_delta = self._new_door_is_behind_corridor_progress(
                center, pose)
            paired_opposite = self._is_confirmed_paired_opposite_side(
                door_station, stable_side)
            # The same strong-evidence rule is used while travelling.  The
            # candidate is recentered below rather than entered directly, so
            # it receives a fresh portal preflight from the corridor side.
            recent_exit_opposite = bool(
                rearward and recent_exit_opposite_side is not None and
                stable_side != recent_exit_opposite_side and
                directed_delta >= -self.room_exit_opposite_door_revisit_radius and
                self.last_room_exit_pose is not None and
                abs(door_station - self._corridor_station(
                    self.last_room_exit_pose)) <=
                self.room_exit_opposite_door_revisit_radius and
                bool(candidate.get("confirmed")))
            # Keep this travelling/preemption path consistent with the
            # regular local-door planner above.  The accepted recent-exit
            # case is recentered below before any room-entry command is sent.
            allow_paired_rearward = bool(
                recent_exit_opposite or
                (rearward and
                 directed_delta >= -self.paired_opposite_maximum_backtrack and
                 paired_opposite and
                 self._rearward_branch_is_paired_opposite(
                     door_station, stable_side,
                     self._corridor_station(pose))))
            terminal_return_recovery = bool(
                self._terminal_return_room_supplement_active() or
                self._missing_room_retrace_active())
            if rearward and not (allow_paired_rearward or
                                 terminal_return_recovery):
                self.corridor_sweep_history.append({
                    "event": "LOCAL_DOOR_PREEMPTION_REJECTED_BEHIND_PROGRESS",
                    "elapsed_sec": round(self.elapsed(), 3),
                    "candidate_id": candidate.get("candidate_id"),
                    "door_center": list(center),
                    "directed_delta_m": round(directed_delta, 3),
                    "maximum_lookbehind_m": self.room_new_door_maximum_lookbehind,
                })
                continue
            if allow_paired_rearward:
                self.corridor_sweep_history.append({
                    "event": ("LOCAL_DOOR_PREEMPTION_RECENT_EXIT_OPPOSITE_ALLOWED"
                              if recent_exit_opposite else
                              "LOCAL_DOOR_PREEMPTION_PAIRED_OPPOSITE_ALLOWED"),
                    "elapsed_sec": round(self.elapsed(), 3),
                    "candidate_id": candidate.get("candidate_id"),
                    "door_center": list(center),
                    "directed_delta_m": round(directed_delta, 3),
                })
            if (len(point) >= 2 and candidate.get("confirmed") and
                    candidate.get("astar_reachable") and
                    candidate.get("scan_lite_safe") and
                    float(candidate.get("width_m", math.inf)) <=
                    self.local_door_maximum_semantic_width and
                    self._local_door_has_room_area(candidate) and
                    abs(float(np.dot(inward, axis))) <=
                    self.local_door_maximum_corridor_alignment and
                    0.45 <= abs(float(np.dot(relative, normal))) <= 3.20 and
                    abs(float(np.dot(relative, axis))) <= 3.50 and
                    not self.room_scheduler.goal_in_visited_room(
                        (float(point[0]), float(point[1]))) and
                    not self._visited_component_for_corridor_goal(
                        point, pose, grid).get("matched")):
                if recent_exit_opposite:
                    # Never take an unvalidated diagonal shortcut from a
                    # moving corridor goal into a recently exposed doorway.
                    # First recenter at the detector's known-free corridor
                    # side, then run the ordinary portal preflight there.
                    pending = dict(candidate)
                    pending["stored_elapsed"] = self.elapsed()
                    pending["paired_opposite_recenter_only"] = True
                    self.pending_paired_opposite_door = pending
                    self.corridor_sweep_history.append({
                        "event": "RECENT_EXIT_OPPOSITE_DOOR_RECENTER_QUEUED",
                        "elapsed_sec": round(self.elapsed(), 3),
                        "candidate_id": candidate.get("candidate_id"),
                        "corridor_side": candidate.get("corridor_side"),
                        "confirmation_count": candidate.get("confirmation_count"),
                    })
                    return pending
                # Do not cancel a healthy corridor goal merely because the
                # external detector likes an aperture.  The room scheduler's
                # stricter free-cell ENTRY selection and mandatory portal
                # preflight must succeed first.  Previously the cancellation
                # happened before this check, producing a cancel/reject loop
                # and long zero-velocity intervals.
                activated = self.room_scheduler.consider_local_door_candidate(
                    grid, (float(pose[0]), float(pose[1])),
                    candidate, self.elapsed())
                if activated is None:
                    if allow_paired_rearward:
                        # The detector has a confirmed A*/SCAN-safe corridor
                        # side point, but a direct long return to the portal
                        # is too stale to pass entry preflight.  Cancel the
                        # current forward goal and first recenter to that
                        # known-free point; the next loop re-runs the normal
                        # doorway preflight from close range.
                        pending = dict(candidate)
                        pending["stored_elapsed"] = self.elapsed()
                        self.pending_paired_opposite_door = pending
                        pending["paired_opposite_recenter_only"] = True
                        self.corridor_sweep_history.append({
                            "event": ("RECENT_EXIT_OPPOSITE_DOOR_RECENTER_QUEUED"
                                      if recent_exit_opposite else
                                      "PAIRED_OPPOSITE_DOOR_RECENTER_QUEUED"),
                            "elapsed_sec": round(self.elapsed(), 3),
                            "candidate_id": candidate.get("candidate_id"),
                            "corridor_side": candidate.get("corridor_side"),
                        })
                        return pending
                    self.corridor_sweep_history.append({
                        "event":
                            "LOCAL_DOOR_PREEMPTION_WITHHELD_UNSAFE_ENTRY",
                        "elapsed_sec": round(self.elapsed(), 3),
                        "candidate_id": candidate.get("candidate_id"),
                        "door_center": candidate.get("door_center"),
                        "entry_goal": candidate.get("entry_goal"),
                        "reason":
                            "room_scheduler_entry_or_portal_preflight_failed",
                    })
                    continue
                validated = dict(candidate)
                validated["preemption_entry_validated"] = True
                validated["activated_door_id"] = activated.door_id
                return validated
        return None

    def _active_exploration_debt(self):
        with self.lock:
            pose = self.pose
        return bool(
            self.room_scheduler.active_door is not None or
            self.pending_paired_opposite_door is not None or
            self._recent_exit_needs_opposite_check(pose) or
            self.depth_breadth.entry_return_pending or
            self.branch_scheduler.active_branch_id is not None or
            self.region_stall_backtrack_pending)

    def _termination_evidence(self):
        now = self.elapsed()
        with self.lock:
            door = dict(self.local_door_status or {})
            entry = dict(self.local_entry_status or {})
            detector_update = self.last_local_detector_update
            door_time = self.last_local_door_received
            entry_time = self.last_local_entry_received
            rescan_time = self.last_local_rescan_completed
        detector_fresh = bool(door and now - detector_update < 8.0)
        unknown = float(door.get("local_unknown_area_m2", math.inf))
        side_openings = int(door.get("side_opening_count", 0))
        active_debt = self._active_exploration_debt()
        checks = {
            "global_reachable_frontier_absent": True,
            "local_detector_fresh": detector_fresh,
            "no_local_side_opening": detector_fresh and side_openings == 0,
            "local_unknown_below_threshold": (
                detector_fresh and unknown <= self.termination_unknown_area_threshold),
            "no_new_local_door_for_20s": (
                now - door_time >= self.local_candidate_recent_window),
            "no_new_room_entry_for_20s": (
                now - entry_time >= self.local_candidate_recent_window),
            "no_active_region_or_debt": not active_debt,
            "local_rescan_completed": rescan_time > -math.inf,
            "maximum_duration_not_misread": now < self.maximum_duration,
        }
        eligible = all(checks.values())
        return eligible, checks, {
            "local_unknown_area_m2": unknown if math.isfinite(unknown) else None,
            "side_opening_count": side_openings,
            "seconds_since_local_door": (
                now - door_time if math.isfinite(door_time) else None),
            "seconds_since_room_entry": (
                now - entry_time if math.isfinite(entry_time) else None),
            "last_rescan_elapsed_sec": (
                rescan_time if math.isfinite(rescan_time) else None),
        }

    def _append_termination_decision(self, action, checks, evidence):
        with self.lock:
            pose = self.pose
        record = {
            "timestamp": time.time(), "elapsed_sec": round(self.elapsed(), 3),
            "robot_pose": None if pose is None else {
                "x": pose[0], "y": pose[1], "yaw": pose[3]},
            "trigger": "NO_VALID_FRONTIER", "action": action,
            "checks": checks, "evidence": evidence,
        }
        self.termination_decisions.append(record)
        path = os.path.join(
            self.output_dir, "logs", "frontier_termination_decision.jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with open(path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
        except OSError as error:
            rospy.logwarn("Could not write frontier termination decision: %s", error)

    def _perform_local_rescan(self, angle_rad=None, angular_speed=None,
                              reason="frontier_exhausted_local_geometry_refresh",
                              visual_sweep=False, room_id=None, direction=1.0):
        if (not visual_sweep and self.elapsed() - self.last_local_rescan_requested <
                self.local_rescan_cooldown):
            # R28: 冷却期 0.25s 空转会让调用处形成 214s 无进展死循环。
            # 改为等待冷却期结束(上限 5s/次),给跌倒恢复/地图更新留窗口。
            wait = min(
                self.local_rescan_cooldown -
                (self.elapsed() - self.last_local_rescan_requested), 5.0)
            time.sleep(max(0.05, wait))
            return False
        self.local_rescan_count += 1
        request_id = "rescan-%d" % self.local_rescan_count
        angle = (self.local_rescan_angle if angle_rad is None else
                 max(0.1, float(angle_rad)))
        speed = (self.local_rescan_speed if angular_speed is None else
                 max(0.05, float(angular_speed)))
        before_grid = self.grid_update_count
        before_results = len(self.local_rescan_results)
        self.last_local_rescan_requested = self.elapsed()
        if (visual_sweep and
                not self._wait_for_visual_sweep_settle(room_id=room_id)):
            return False
        self._set_state("LOCAL_RESCAN", reason)
        action_limit = (self.room_visual_sweep_maximum_seconds
                        if visual_sweep else
                        self.local_rescan_maximum_seconds)
        self.local_rescan_pub.publish(String(data=json.dumps({
            "request_id": request_id, "angle_rad": angle,
            "angular_speed": speed,
            "visual_sweep": bool(visual_sweep),
            "full_visual_sweep": bool(visual_sweep and angle > math.pi + .01),
            "room_id": room_id,
            "direction": -1.0 if float(direction) < 0.0 else 1.0,
            "timeout_sec": min(
                angle / speed + 2.0,
                action_limit),
        }, sort_keys=True)))
        deadline = min(
            time.monotonic() + action_limit + 0.5,
            self.started + self.maximum_duration)
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            with self.lock:
                results = list(self.local_rescan_results[before_results:])
            match = next((item for item in results
                          if item.get("request_id") == request_id), None)
            if match is not None:
                # Give the mapper/detector at least one post-rotation update.
                update_deadline = time.monotonic() + (.45 if visual_sweep else 2.0)
                while (not rospy.is_shutdown() and
                       time.monotonic() < update_deadline and
                       self.grid_update_count <= before_grid):
                    time.sleep(0.05)
                return bool(match.get("success"))
            time.sleep(0.05)
        return False

    def _home_reached(self, pose=None):
        with self.lock:
            current = pose if pose is not None else self.pose
            start = self.mission_start_pose
        return bool(current is not None and start is not None and math.hypot(
            float(current[0]) - float(start[0]),
            float(current[1]) - float(start[1])) <= self.mission_home_tolerance)

    def _stair_wait_target(self):
        """Use the first online-validated G1 waypoint, else the start pose."""
        with self.lock:
            target = (self.stair_wait_anchor if self.stair_wait_anchor is not None
                      else self.mission_start_pose)
        return tuple(target) if target is not None else None

    def _stair_wait_reached(self, pose=None):
        with self.lock:
            current = pose if pose is not None else self.pose
        target = self._stair_wait_target()
        return bool(current is not None and target is not None and math.hypot(
            float(current[0]) - float(target[0]),
            float(current[1]) - float(target[1])) <= self.stair_wait_tolerance)

    def _plan_mission_return_goal(self):
        """Return to the online G1/stair staging point after all rooms."""
        with self.lock:
            pose = self.pose
        target = self._stair_wait_target()
        if pose is None or target is None:
            return None
        return {
            "position": [float(target[0]), float(target[1]), float(pose[2])],
            "information_gain": 0.0, "score": 0.0,
            "source": "mission_return_home",
            "scheduler_phase": "MISSION_RETURN_HOME",
            "execution_timeout_sec": max(
                30.0, min(self.goal_timeout,
                          self._stair_return_remaining())),
        }

    def _plan_corridor_exit_handoff_goal(self):
        """Return to the traversed lobby/corridor entrance for stair handoff.

        The corridor station origin is *not* its entrance: corridor geometry
        is only established after the acquisition goals have already moved
        several metres down the corridor.  Returning to that origin left the
        robot behind the room/stair-core partition, from where a straight
        truth-guided stair approach collided with the wall.  The first
        successful acquisition waypoint is both footprint-validated and in
        the open lobby beside the stair core, so use the existing stair-wait
        target captured there (falling back to the mission start pose).

        This target is learned from the traversed first-floor route and does
        not encode the stair location.  The optional truth-entry guide owns
        only the short motion from this safe handoff point to the first tread.
        """
        with self.lock:
            pose, grid = self.pose, self.grid
        target = self._stair_wait_target()
        if pose is None or target is None:
            return self._plan_mission_return_goal()
        preferred = []
        axis = (self.corridor_station_axis if self.corridor_station_axis is not None
                else self.corridor_axis)
        if axis is not None:
            axis = np.asarray(axis, dtype=float)
            norm = float(np.linalg.norm(axis))
            if norm > 1e-9:
                axis /= norm
                for distance in (.30,-.30,.60,-.60,.90,-.90,1.20,-1.20,
                                 1.80,-1.80,2.40,-2.40):
                    preferred.append((
                        float(target[0])+distance*float(axis[0]),
                        float(target[1])+distance*float(axis[1])))
        # Always prefer the reverse of the physically traversed corridor for
        # a long post-room return.  The former direct A* chord exposed one
        # 24 m waypoint to accumulated FAST-LIO scale/loop drift; run29 drove
        # past the physical lobby while its map pose still reported x=6 m.
        # Breadcrumbs bound each command to a recently occupied centreline
        # pose and give the truth stair gate repeated takeover opportunities.
        breadcrumbs = self._stair_corridor_return_breadcrumbs(target, pose)
        if len(breadcrumbs) >= 2:
            path = [(float(pose[0]), float(pose[1]))] + [
                (float(point[0]), float(point[1])) for point in breadcrumbs]
            self.corridor_sweep_history.append({
                "event": "STAIR_RETURN_VERIFIED_BREADCRUMBS_SELECTED",
                "elapsed_sec": round(self.elapsed(), 3),
                "target": list(target[:2]),
                "selection_policy": "always_for_long_return",
                "waypoint_count": len(breadcrumbs),
            })
            return {
                "position": [float(target[0]), float(target[1]),
                             float(pose[2])],
                "information_gain": 0.0, "score": 0.0,
                "source": "stair_corridor_exit_handoff",
                "scheduler_phase": "STAIR_CORRIDOR_EXIT_HANDOFF",
                "execution_timeout_sec": max(
                    20.0, min(self.goal_timeout,
                              self._stair_return_remaining())),
                "verified_trajectory_backtrack": True,
                "breadcrumb_fallback": False,
                "_preplanned_path_result": {
                    "success": True,
                    "reason": "verified_corridor_breadcrumb_return",
                    "path": path, "execution_waypoints": breadcrumbs},
            }
        selection = (nearest_reachable_goal(
            grid,(float(pose[0]),float(pose[1])),
            (float(target[0]),float(target[1])),min(self.clearance,.20),
            self.reached_tolerance,self.stair_return_relocation_radius,128,
            preferred) if grid is not None else
            {"success":False,"reason":"map_unavailable"})
        if selection.get("success"):
            selected=selection.get("target") or target
            displacement=float(selection.get("displacement",0.0))
            if displacement > 1e-6:
                self.corridor_sweep_history.append({
                    "event":"STAIR_RETURN_TARGET_RELOCATED",
                    "elapsed_sec":round(self.elapsed(),3),
                    "original_target":list(target[:2]),
                    "selected_target":list(selected[:2]),
                    "displacement_m":round(displacement,3),
                    "candidates_tested":selection.get("candidates_tested",0),
                })
            return {
                "position":[float(selected[0]),float(selected[1]),float(pose[2])],
                "information_gain":0.0,"score":0.0,
                "source":"stair_corridor_exit_handoff",
                "scheduler_phase":"STAIR_CORRIDOR_EXIT_HANDOFF",
                "execution_timeout_sec":max(
                    20.0,min(self.goal_timeout,
                             self._stair_return_remaining())),
                "stair_return_original_target":list(target[:2]),
                "stair_return_relocated":displacement > 1e-6,
                "planning_clearance_m":min(self.clearance,.20),
            }
        # The historical entrance can be falsely occupied as an entire block
        # after vertical/map drift.  In that case no nearby free goal exists,
        # although the robot physically traversed the straight corridor.
        # Recover over centreline poses recorded on that actual traversal.
        breadcrumbs=self._stair_corridor_return_breadcrumbs(target,pose)
        if breadcrumbs:
            path=[(float(pose[0]),float(pose[1]))]+[
                (float(point[0]),float(point[1])) for point in breadcrumbs]
            self.corridor_sweep_history.append({
                "event":"STAIR_RETURN_VERIFIED_BREADCRUMBS_SELECTED",
                "elapsed_sec":round(self.elapsed(),3),
                "target":list(target[:2]),
                "map_failure":selection.get("reason"),
                "waypoint_count":len(breadcrumbs),
            })
            return {
                "position":[float(target[0]),float(target[1]),float(pose[2])],
                "information_gain":0.0,"score":0.0,
                "source":"stair_corridor_exit_handoff",
                "scheduler_phase":"STAIR_CORRIDOR_EXIT_HANDOFF",
                "execution_timeout_sec":max(
                    20.0,min(self.goal_timeout,
                             self._stair_return_remaining())),
                "verified_trajectory_backtrack":True,
                "breadcrumb_fallback":True,
                "_preplanned_path_result":{
                    "success":True,"reason":"verified_corridor_breadcrumb_return",
                    "path":path,"execution_waypoints":breadcrumbs},
            }
        return {
            "position": [float(target[0]), float(target[1]), float(pose[2])],
            "information_gain": 0.0, "score": 0.0,
            "source": "stair_corridor_exit_handoff",
            "scheduler_phase": "STAIR_CORRIDOR_EXIT_HANDOFF",
            "execution_timeout_sec": max(
                20.0, min(self.goal_timeout,
                            self._stair_return_remaining())),
        }

    def _stair_corridor_return_breadcrumbs(self,target,pose):
        """Build a monotonic reverse route from physically traversed centreline poses."""
        with self.lock:
            trajectory=list(self.trajectory)
        axis=(self.corridor_station_axis if self.corridor_station_axis is not None
              else self.corridor_axis)
        origin=(self.corridor_station_origin if self.corridor_station_origin is not None
                else target)
        if axis is None or not trajectory:
            return self._region_return_breadcrumbs(
                target,(float(pose[0]),float(pose[1])),
                self.stair_return_breadcrumb_spacing)
        axis=np.asarray(axis,dtype=float); norm=float(np.linalg.norm(axis))
        if norm < 1e-9:
            return []
        axis/=norm; normal=np.asarray([-axis[1],axis[0]],dtype=float)
        origin=np.asarray(origin[:2],dtype=float)
        current=np.asarray(pose[:2],dtype=float)
        destination=np.asarray(target[:2],dtype=float)
        current_station=float(np.dot(current-origin,axis))
        target_station=float(np.dot(destination-origin,axis))
        direction=1.0 if target_station>current_station else -1.0
        low=min(current_station,target_station)-.25
        high=max(current_station,target_station)+.25
        candidates=[]
        for sample in trajectory:
            if len(sample)<3:
                continue
            point=np.asarray([float(sample[1]),float(sample[2])],dtype=float)
            station=float(np.dot(point-origin,axis))
            lateral=abs(float(np.dot(point-origin,normal)))
            if (low<=station<=high and
                    lateral<=self.stair_return_centerline_tolerance):
                candidates.append((station,lateral,point))
        spacing=max(.45,self.stair_return_breadcrumb_spacing)
        desired=current_station+direction*spacing
        anchors=[]; used=set()
        while ((direction>0 and desired<target_station) or
               (direction<0 and desired>target_station)):
            ranked=sorted(candidates,key=lambda item:
                          (abs(item[0]-desired)+.35*item[1],item[1]))
            chosen=next((item for item in ranked
                         if id(item[2]) not in used and
                         abs(item[0]-desired)<=spacing),None)
            if chosen is not None:
                point=(float(chosen[2][0]),float(chosen[2][1]))
                if (not anchors or math.hypot(point[0]-anchors[-1][0],
                                               point[1]-anchors[-1][1])>=.35):
                    anchors.append(point); used.add(id(chosen[2]))
            desired+=direction*spacing
        final=(float(target[0]),float(target[1]))
        if not anchors or math.hypot(anchors[-1][0]-final[0],
                                     anchors[-1][1]-final[1])>.15:
            anchors.append(final)
        return [[point[0],point[1]] for point in anchors]

    def _plan_stair_return_escape_goal(self):
        """Make one short centreline retreat before the long entrance return."""
        with self.lock:
            pose = self.pose
            axis = (None if self.corridor_axis is None else
                    np.asarray(self.corridor_axis, dtype=float).copy())
        if pose is None or axis is None:
            return None
        norm = float(np.linalg.norm(axis))
        if norm < 1e-6:
            return None
        axis /= norm
        target = (float(pose[0]) - 1.8 * float(axis[0]),
                  float(pose[1]) - 1.8 * float(axis[1]))
        return {"position": [target[0], target[1], float(pose[2])],
                "information_gain": 0.0, "score": 0.0,
                "source": "stair_corridor_escape",
                "scheduler_phase": "STAIR_CORRIDOR_ESCAPE",
                "verified_trajectory_backtrack": True,
                "_preplanned_path_result": {"success": True,
                    "reason": "terminal_wall_centreline_escape",
                    "path": [(float(pose[0]), float(pose[1])), target],
                    "execution_waypoints": [target]}}

    def _plan_corridor_door_goal(self):
        """Give a stable forward side-door candidate priority over FUEL."""
        if (not self.corridor_door_detection_armed or
                not self.room_scheduler.config.enabled or
                self.room_scheduler.active_door is not None):
            return None
        with self.lock:
            pose, grid = self.pose, self.grid
            observed_update = self.grid_update_count
        # ROOM_EXIT is confirmed only 0.4 m beyond the door plane.  That is a
        # safe crossing, but it can still leave the robot hugging one corridor
        # wall, where the opening behind it makes corridor classification fail
        # and FUEL immediately drives past the opposite doorway.  Restore the
        # saved pre-entry centreline before either scanning doors or allowing a
        # forward corridor goal.
        if (self.room_scheduler.state == "CORRIDOR_RESUME" and
                self.corridor_axis is not None and
                self.corridor_anchor is not None):
            exited_rooms = sum(
                bool(door.visited)
                for door in self.room_scheduler.detector.doors)
            pair_complete_forward_merge = bool(
                pose is not None and
                self.corridor_far_pair_search_advance > 0.05 and
                exited_rooms == self.corridor_far_pair_search_after_exits and
                exited_rooms < self.room_target_count and
                not self._recent_exit_needs_opposite_check(pose))
            if pair_complete_forward_merge:
                self.room_scheduler.abandon_corridor_resume(
                    self.elapsed(), "pair_complete_forward_resume_merged")
                self.corridor_resume_failures = 0
                self.corridor_sweep_history.append({
                    "event": "CORRIDOR_RESUME_MERGED_INTO_FORWARD_SEARCH",
                    "elapsed_sec": round(self.elapsed(), 3),
                    "exited_rooms": int(exited_rooms),
                    "target_rooms": int(self.room_target_count),
                    "policy": "pair_complete_continuous_astar_transit",
                })
                return None
            axis = np.asarray(self.corridor_axis, dtype=float)
            anchor = np.asarray(self.corridor_anchor, dtype=float)
            current = np.asarray([float(pose[0]), float(pose[1])], dtype=float)
            centre = anchor + np.dot(current - anchor, axis) * axis
            lateral_error = float(np.linalg.norm(current - centre))
            if lateral_error > 0.30:
                if self.corridor_resume_failures >= self.corridor_resume_failure_limit:
                    self.room_scheduler.abandon_corridor_resume(
                        self.elapsed(), "corridor_recenter_failure_limit")
                    self.corridor_sweep_history.append({
                        "event": "CORRIDOR_RESUME_RELEASED",
                        "elapsed_sec": round(self.elapsed(), 3),
                        "reason": "recenter_failure_limit",
                        "failure_count": self.corridor_resume_failures,
                        "lateral_error_m": lateral_error,
                    })
                    self.corridor_resume_failures = 0
                else:
                    return {
                        "position": [float(centre[0]), float(centre[1]), pose[2]],
                        "information_gain": 0.0,
                        "score": 0.0,
                        "source": "corridor_resume_centerline",
                        "scheduler_phase": "CORRIDOR_RESUME",
                        "corridor_resume_lateral_error_m": lateral_error,
                    }
            else:
                self.room_scheduler.abandon_corridor_resume(
                    self.elapsed(), "corridor_centerline_reached")
                self.corridor_resume_failures = 0
        context = self._door_search_corridor_context(
            pose, grid, "corridor_side_scan")
        if context is None:
            return None
        if not self._on_station_corridor_centerline(pose):
            return None
        if not self._at_corridor_forward_high_water(pose, context):
            self.corridor_sweep_history.append({
                "event": "DOOR_TAKEOVER_REJECTED_DURING_BACKTRACK",
                "elapsed_sec": round(self.elapsed(), 3),
                "station": self._corridor_station(pose),
                "forward_high_water": self.corridor_forward_station_high_water,
            })
            return None
        if (self.corridor_station_axis is None or
                self.corridor_station_origin is None):
            return None
        # Confirmation is local-geometry based. Never require reaching a
        # particular global corridor station or accumulated y-like progress.
        candidate = self.room_scheduler.detector.corridor_side_candidate(
            grid, (pose[0], pose[1]), context["heading"], self.elapsed(),
            lookbehind=self.room_new_door_maximum_lookbehind)
        if candidate is None:
            return None
        if self._reject_initial_lobby_throat(
                candidate.center, getattr(candidate, "candidate_id", None),
                "corridor_side_scan"):
            return None
        if not self._door_centerline_lateral_admissible(
                candidate.center, "corridor_side_scan",
                getattr(candidate, "candidate_id", None)):
            return None
        if math.isfinite(self.corridor_side_candidate_max_forward_station):
            sign = (self.corridor_forward_station_sign
                    if self.corridor_forward_station_sign is not None else 1.0)
            directed_delta = float(sign) * (
                self._corridor_station(candidate.center) -
                self._corridor_station(pose))
            if (directed_delta >
                    self.corridor_side_candidate_max_forward_station):
                self.corridor_sweep_history.append({
                    "event":
                        "DOOR_REJECTED_BEYOND_FORWARD_OBSERVATION_WINDOW",
                    "elapsed_sec": round(self.elapsed(), 3),
                    "candidate_center": list(candidate.center),
                    "directed_forward_delta_m": round(directed_delta, 3),
                    "maximum_forward_delta_m":
                        self.corridor_side_candidate_max_forward_station,
                    "source": "corridor_side_scan",
                })
                return None
        # This direct local-door path must apply the same terminal-wall
        # rejection as the planned-path interceptor below.  Otherwise a
        # rolling-map gap at the end of a corridor can bypass that check and
        # be committed as a room entry.
        if not doorway_is_lateral_to_corridor(
                candidate.normal_direction, context.get("axis"),
                self.room_door_maximum_corridor_alignment):
            self.corridor_sweep_history.append({
                "event": "DOOR_GEOMETRY_REJECTED_FORWARD_TERMINAL_LOCAL_SCAN",
                "elapsed_sec": round(self.elapsed(), 3),
                "candidate_center": list(candidate.center),
                "door_normal_direction": float(candidate.normal_direction),
                "corridor_axis": list(context.get("axis", [])),
                "reason": "aperture_normal_aligned_with_corridor",
            })
            return None
        # A local side-opening candidate is actionable immediately.  Do not
        # defer it because it is on the same side as the previous exit or wait
        # for another map confirmation; those policies caused real room 0/1
        # openings to be skipped before the robot ever attempted entry.
        confirmed = self.room_scheduler.consider_corridor_door(
            grid, (pose[0], pose[1]), context["heading"], self.elapsed(),
            lookbehind=self.room_new_door_maximum_lookbehind)
        if confirmed is None:
            return None
        goal = self.room_scheduler.next_goal(
            grid, (pose[0], pose[1]), self.elapsed())
        if goal is not None:
            goal["position"][2] = pose[2]
            goal["scheduler_phase"] = "ROOM_ENTRY"
            goal["door_takeover_source"] = "corridor_side_scan"
            rospy.loginfo("Proactive side-door takeover: %s center=(%.2f, %.2f)",
                          confirmed.door_id, confirmed.center[0], confirmed.center[1])
        return goal

    def _intercept_room_bound_path(self, cycle, goal, path_result):
        """Replace a generic room-bound path with a confirmed portal commit.

        Confirmation is intentionally made against three distinct occupancy
        projections.  Until that succeeds the original FUEL target is never
        sent to locomotion, so a distant room frontier cannot shortcut a wall.
        """
        if (not self.corridor_door_detection_armed or
                not self.room_scheduler.config.enabled or
                not self.enable_planned_room_path_intercept or
                self.room_scheduler.active_door is not None or
                not path_result.get("success") or
                str(goal.get("source", "")).startswith("lightweight_room_") or
                goal.get("source") == "mission_return_home"):
            return goal, path_result
        with self.lock:
            grid = self.grid
            pose = self.pose
            observed_update = self.grid_update_count
        corridor = self._corridor_context(pose, grid)
        if (pose is None or
                not self._on_established_corridor_centerline(pose, corridor) or
                self.corridor_station_axis is None or
                self.corridor_station_origin is None):
            return goal, path_result
        # The path/aperture evidence below is local and remains valid even
        # when FAST-LIO longitudinal scale has drifted.
        candidate = self.room_scheduler.detector.planned_candidate(
            grid, path_result.get("path", []), self.elapsed()) if grid is not None else None
        if candidate is None:
            return goal, path_result
        if self._reject_initial_lobby_throat(
                candidate.center, getattr(candidate, "candidate_id", None),
                "planned_path_intercept"):
            return goal, path_result
        if not self._door_centerline_lateral_admissible(
                candidate.center, "planned_path_intercept",
                getattr(candidate, "candidate_id", None)):
            return goal, path_result

        # A forward aperture at the corridor terminal wall can look like a
        # wide opening in a rolling occupancy grid.  It is not a room door:
        # a real side door's normal must be approximately perpendicular to the
        # corridor axis.  The opt1 run activated a false door at x≈29 m whose
        # normal was parallel to the corridor, then spent the rest of the run
        # exploring the corridor end.  Reject this before room activation;
        # the original forward FUEL path remains valid and continues through
        # the corridor instead.
        corridor_context = self._corridor_context(pose, grid)
        if (corridor_context is not None and
                not doorway_is_lateral_to_corridor(
                    candidate.normal_direction,
                    corridor_context.get("axis"),
                    self.room_door_maximum_corridor_alignment)):
            self.corridor_sweep_history.append({
                "event": "DOOR_GEOMETRY_REJECTED_FORWARD_TERMINAL",
                "elapsed_sec": round(self.elapsed(), 3),
                "candidate_center": list(candidate.center),
                "door_normal_direction": float(candidate.normal_direction),
                "corridor_axis": list(corridor_context.get("axis", [])),
                "reason": "aperture_normal_aligned_with_corridor",
            })
            return goal, path_result

        event_start = len(self.room_scheduler.events)
        confirmed = None
        required = self.room_scheduler.config.candidate_confirmation_count
        for confirmation_index in range(required):
            if confirmation_index:
                deadline = time.monotonic() + 2.0
                while not rospy.is_shutdown() and time.monotonic() < deadline:
                    with self.lock:
                        if self.grid_update_count > observed_update:
                            grid = self.grid
                            observed_update = self.grid_update_count
                            break
                    time.sleep(0.03)
                else:
                    break
            confirmed = self.room_scheduler.consider_planned_path(
                grid, path_result.get("path", []), self.elapsed())
            if confirmed is not None:
                break
        if confirmed is None:
            rejection = next((
                event for event in reversed(
                    self.room_scheduler.events[event_start:])
                if event.get("event") == "DOOR_CANDIDATE_REJECTED"), None)
            if rejection is not None:
                # A fully observed but semantically rejected aperture is not
                # the same as an unstable candidate.  The original FUEL path
                # is already known-free A* and will still pass SCAN-lite, so
                # do not spend the rest of the mission holding the same goal.
                self.corridor_sweep_history.append({
                    "event": "DOOR_GEOMETRY_REJECTED_FUEL_PASSTHROUGH",
                    "elapsed_sec": round(self.elapsed(), 3),
                    "cycle": cycle,
                    "candidate_center": list(candidate.center),
                    "reason": rejection.get("reason"),
                    "fuel_goal": goal.get("position"),
                })
                goal["door_geometry_rejected_passthrough"] = True
                goal["door_rejection_reason"] = rejection.get("reason")
                return goal, path_result
            rospy.logwarn("Door candidate on cycle %d was not stable across %d map updates; "
                          "holding the room-bound FUEL goal", cycle, required)
            return goal, {
                "success": False, "reason": "door_candidate_unconfirmed",
                "path": path_result.get("path", []),
                "candidate_held": True,
            }

        with self.lock:
            pose, grid = self.pose, self.grid
        portal_goal = self.room_scheduler.next_goal(
            grid, (pose[0], pose[1]), self.elapsed())
        if portal_goal is None:
            return goal, {"success": False,
                          "reason": "confirmed_door_without_safe_g1", "path": []}
        portal_goal["position"][2] = pose[2]
        portal_goal["scheduler_phase"] = "ROOM_ENTRY"
        portal_goal["intercepted_goal"] = goal.get("position")
        portal_goal["intercepted_goal_source"] = goal.get("source", "fuel_frontier")
        rospy.loginfo("Door takeover: cycle=%d door=%s FUEL=(%.2f, %.2f) ENTRY=(%.2f, %.2f)",
                      cycle, confirmed.door_id,
                      float(goal["position"][0]), float(goal["position"][1]),
                      float(portal_goal["position"][0]),
                      float(portal_goal["position"][1]))
        return portal_goal, self._plan_path(cycle, portal_goal)

    def _plan_path(self, cycle, goal):
        self._set_state("PLAN_PATH")
        with self.lock:
            pose, grid = self.pose, self.grid
        target = goal.get("position") or []
        if len(target) < 2 or grid is None or pose is None:
            result = {"success": False, "reason": "invalid_planner_goal", "path": []}
        else:
            mandatory = goal.get("mandatory_portal_waypoints") or []
            planning_clearance = float(
                goal.get(
                    "portal_clearance_m" if mandatory else
                    "planning_clearance_m",
                    self.clearance))
            segment_targets = [
                (float(point[0]), float(point[1])) for point in mandatory
                if isinstance(point, (list, tuple)) and len(point) >= 2]
            if not segment_targets:
                segment_targets = [(float(target[0]), float(target[1]))]
            segment_start = (pose[0], pose[1])
            combined_path = []
            result = None
            # A rolling occupancy projection can lag the live odometry by a
            # cell immediately after a failed doorway preemption.  Permit a
            # blocked start only for corridor recovery/transit goals; portal
            # and room-entry segments remain strictly validated.
            source = str(goal.get("source", ""))
            allow_recovery_start = (not mandatory and source in (
                "corridor_sweep", "corridor_forward_no_frontier",
                "corridor_no_frontier_recovery", "corridor_forward_recovery",
                "corridor_transit", "stair_corridor_exit_handoff"))
            for segment_index, segment_target in enumerate(segment_targets):
                segment = astar_safe_path(
                    grid, segment_start, segment_target, planning_clearance,
                    self.reached_tolerance,
                    allow_blocked_start=allow_recovery_start)
                if not segment.get("success"):
                    result = dict(segment)
                    if mandatory:
                        result["reason"] = "portal_segment_{}_{}".format(
                            segment_index, segment.get("reason", "failed"))
                    break
                points = list(segment.get("path", []))
                combined_path.extend(points if not combined_path else points[1:])
                segment_start = segment_target
            if result is None:
                result = {"success": True, "reason":
                          "portal_path_found" if mandatory else "path_found",
                          "path": combined_path}
            # The 2-D circular inflation is intentionally conservative and can
            # mark the robot's physically occupied start cell as blocked.  Only
            # override that single start check when the live 3-D twin-cylinder
            # footprint confirms the current body pose is collision-free.
            if result.get("reason") in ("start_footprint_blocked",
                                        "portal_segment_0_start_footprint_blocked"):
                footprint = self._scan_check(pose[0], pose[1], pose[3])
                if footprint.map_available and not footprint.occupied_collision:
                    first_target = segment_targets[0]
                    first = astar_safe_path(
                        grid, (pose[0], pose[1]), first_target,
                        planning_clearance, self.reached_tolerance,
                        allow_blocked_start=True)
                    if first.get("success"):
                        combined_path = list(first.get("path", []))
                        segment_start = first_target
                        result = None
                        for segment_index, segment_target in enumerate(
                                segment_targets[1:], 1):
                            segment = astar_safe_path(
                                grid, segment_start, segment_target,
                                planning_clearance, self.reached_tolerance)
                            if not segment.get("success"):
                                result = dict(segment)
                                result["reason"] = "portal_segment_{}_{}".format(
                                    segment_index, segment.get("reason", "failed"))
                                break
                            combined_path.extend(segment.get("path", [])[1:])
                            segment_start = segment_target
                        if result is None:
                            result = {"success": True,
                                      "reason": "portal_path_found_start_override",
                                      "path": combined_path}
                    else:
                        result = first
                    result["start_override_3d_verified"] = True
                    result["start_override_status"] = footprint.status
        record = {
            "cycle": cycle, "elapsed_sec": round(self.elapsed(), 3),
            "goal": target, "success": bool(result["success"]),
            "reason": result["reason"], "clearance_m": planning_clearance
            if len(target) >= 2 and grid is not None and pose is not None
            else self.clearance,
            "start_override_3d_verified": bool(
                result.get("start_override_3d_verified", False)),
            "path": [[round(p[0], 4), round(p[1], 4)] for p in result["path"]],
        }
        self.path_history.append(record)
        if result["success"]:
            self.astar_success_count += 1
        self._write_all_logs()
        return result

    def _publish_speed_for_goal(self, goal):
        """Set a state-aware limit through Goal Executor (RL remains owner)."""
        source = str(goal.get("source", ""))
        room_active = self.room_scheduler.active_door is not None
        room_source = source.startswith("lightweight_room") or source in (
            "local_doorway_detector", "corridor_side_coverage_debt",
            "corridor_remembered_side_branch", "corridor_side_branch_deepen",
            "region_coverage_entry_return", "region_stall_backtrack")
        room_role = str(goal.get("room_role", "")).upper()
        if room_role == "EXIT" or source.startswith("lightweight_room_exit"):
            speed, lateral_speed, mode = (self.room_exit_speed,
                                          self.room_exit_lateral_speed_limit,
                                          "ROOM_EXIT_SAFE")
        elif room_active or room_source:
            speed, lateral_speed, mode = (self.room_entry_speed,
                                          self.room_lateral_speed_limit,
                                          "ROOM_OR_DOOR")
        elif source in ("corridor_resume_centerline", "corridor_forward_no_frontier",
                        "corridor_monotonic_forward_probe",
                        "corridor_transit", "mission_return_home"):
            speed, lateral_speed, mode = (self.corridor_transit_speed,
                                          self.corridor_lateral_speed_limit,
                                          ("MISSION_RETURN_FAST" if
                                           source == "mission_return_home"
                                           else "CORRIDOR_TRANSIT"))
        elif source == "corridor_sweep":
            speed, lateral_speed, mode = (self.corridor_search_speed,
                                          self.corridor_lateral_speed_limit,
                                          "CORRIDOR_SEARCH")
        else:
            speed, lateral_speed, mode = (self.corridor_search_speed,
                                          self.corridor_lateral_speed_limit,
                                          "EXPLORATION_DEFAULT")
        speed = max(0.05, float(speed))
        lateral_speed = max(0.05, float(lateral_speed))
        self.speed_limit_pub.publish(Float32(data=speed))
        self.lateral_speed_limit_pub.publish(Float32(data=lateral_speed))
        # Keep all exploration phases on the launch-time controller gain.
        # Only the final verified return gets a faster approach to the same
        # calibrated velocity cap.
        distance_gain = None
        if source == "mission_return_home":
            distance_gain = self.return_goal_distance_gain
            self.distance_gain_pub.publish(Float32(data=distance_gain))
        self.corridor_sweep_history.append({
            "event": "SPEED_MODE_CHANGED",
            "elapsed_sec": round(self.elapsed(), 3),
            "mode": mode, "speed_limit_mps": speed,
            "lateral_speed_limit_mps": lateral_speed,
            "distance_gain": distance_gain, "source": source,
        })

    def _publish_waypoint(self, point, z, yaw, semantic_deadline=None,
                          allow_local_door_preempt=False):
        message = PoseStamped()
        message.header.stamp = rospy.Time.now()
        message.header.frame_id = self.pose_frame
        message.pose.position.x, message.pose.position.y = point
        message.pose.position.z = z
        message.pose.orientation.z = math.sin(yaw * 0.5)
        message.pose.orientation.w = math.cos(yaw * 0.5)
        with self.lock:
            baseline = len(self.execution_results)
        self.goal_pub.publish(message)
        goal_stamp = float(message.header.stamp.to_sec())

        def matching_result():
            with self.lock:
                candidates = list(self.execution_results[baseline:])
            for candidate in candidates:
                if execution_result_matches(candidate, goal_stamp, point):
                    return dict(candidate)
            return None

        remaining_mission = self._stair_return_remaining()
        deadline = time.monotonic() + min(self.goal_timeout, remaining_mission)
        if semantic_deadline is not None:
            deadline = min(deadline, float(semantic_deadline))
        preempt_candidate = None
        localization_hold_requested = False
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            with self.lock:
                stair_gate_reached = self.stair_truth_return_gate_reached
            if stair_gate_reached:
                self.cancel_goal_pub.publish(String(data=json.dumps({
                    "goal_sequence": int(message.header.seq),
                    "goal_stamp": goal_stamp,
                    "goal_x": float(point[0]), "goal_y": float(point[1]),
                    "reason": "stair_truth_return_gate_reached",
                }, sort_keys=True)))
                return {
                    "success": False,
                    "reason": "stair_truth_return_gate_reached",
                }
            result = matching_result()
            if result is not None:
                if (preempt_candidate is not None and result.get("reason") ==
                        "goal_cancelled_by_manager:confirmed_local_door_preemption"):
                    return {
                        "success": True,
                        "reason": "path_preempted_for_local_door",
                        "preempted_goal_result": result,
                        "local_door_candidate_id":
                            preempt_candidate.get("candidate_id"),
                    }
                # Do not disguise a controller/fall failure as a successful
                # semantic handoff merely because a doorway had triggered the
                # cancellation request.  In particular, this keeps
                # locomotion_not_ready visible to the scheduler.
                return result
            with self.lock:
                registration_healthy = self.fastlio_registration_healthy
                registration_invalid_count = \
                    self.fastlio_registration_invalid_count
            if (self.localization_truth_recovery_enabled and
                    not localization_hold_requested and
                    not registration_healthy and
                    registration_invalid_count >=
                    self.localization_truth_recovery_start_invalid_count):
                localization_hold_requested = True
                self.cancel_goal_pub.publish(String(data=json.dumps({
                    "goal_sequence": int(message.header.seq),
                    "goal_stamp": goal_stamp,
                    "goal_x": float(point[0]),
                    "goal_y": float(point[1]),
                    "reason": "localization_truth_recovery_hold",
                }, sort_keys=True)))
                self.corridor_sweep_history.append({
                    "event": "ACTIVE_WAYPOINT_LOCALIZATION_HOLD_REQUESTED",
                    "elapsed_sec": round(self.elapsed(), 3),
                    "goal_sequence": int(message.header.seq),
                    "goal_stamp": goal_stamp,
                    "invalid_count": registration_invalid_count,
                    "policy":
                        "cancel_stale_path_before_truth_recovery",
                })
                time.sleep(0.01)
            if (allow_local_door_preempt and preempt_candidate is None):
                preempt_candidate = self._local_door_preemption_candidate()
                if preempt_candidate is not None:
                    self.cancel_goal_pub.publish(String(data=json.dumps({
                        "goal_sequence": int(message.header.seq),
                        "goal_stamp": goal_stamp,
                        "goal_x": float(point[0]), "goal_y": float(point[1]),
                        "reason": "confirmed_local_door_preemption",
                    }, sort_keys=True)))
                    self.corridor_sweep_history.append({
                        "event": "CORRIDOR_GOAL_PREEMPTED_FOR_LOCAL_DOOR",
                        "elapsed_sec": round(self.elapsed(), 3),
                        "candidate_id": preempt_candidate.get("candidate_id"),
                        "door_center": preempt_candidate.get("door_center"),
                        "entry_goal": preempt_candidate.get("entry_goal"),
                        "entry_preflight_validated": bool(
                            preempt_candidate.get(
                                "preemption_entry_validated", False)),
                        "activated_door_id": preempt_candidate.get(
                            "activated_door_id"),
                    })
            time.sleep(0.05)
        reason = ("TIME_LIMIT" if self.elapsed() >= self._stair_return_deadline() else
                  "semantic_goal_timeout" if semantic_deadline is not None and
                  time.monotonic() >= semantic_deadline else "goal_executor_timeout")
        self.cancel_goal_pub.publish(String(data=json.dumps({
            "goal_sequence": int(message.header.seq),
            "goal_stamp": float(message.header.stamp.to_sec()),
            "goal_x": float(point[0]), "goal_y": float(point[1]),
            "reason": reason,
        }, sort_keys=True)))
        cancel_deadline = time.monotonic() + 1.0
        while not rospy.is_shutdown() and time.monotonic() < cancel_deadline:
            if matching_result() is not None:
                break
            time.sleep(0.03)
        return {"success": False, "reason": reason}

    def _execute_path(self, path_result, goal, cycle):
        if path_result["reason"] == "already_at_goal":
            return {"success": True, "reason": "already_at_goal", "waypoints": 0}
        self._set_state("EXECUTE_GOAL")
        waypoints = list(path_result.get("execution_waypoints") or
                         execution_waypoints(path_result["path"], self.waypoint_spacing))
        original_waypoint_count = len(waypoints)
        with self.lock:
            execution_pose = self.pose
        waypoints, precompleted = drop_reached_waypoint_prefix(
            waypoints, execution_pose, self.reached_tolerance)
        if not waypoints:
            return {"success": True, "reason": "already_at_path_end",
                    "waypoints": original_waypoint_count,
                    "completed_waypoints": original_waypoint_count,
                    "skipped_reached_prefix": precompleted,
                    "zero_command_stall_recovery_count": 0}
        z = float((goal.get("position") or [0, 0, self.pose[2]])[2])
        semantic_timeout = goal.get("execution_timeout_sec")
        semantic_deadline = (time.monotonic() + float(semantic_timeout)
                             if semantic_timeout is not None else None)
        progressive_timeout = bool(goal.get("progressive_timeout", False))
        progress_timeout = max(
            1.0, float(goal.get("progress_timeout_sec", 8.0)))
        recoveries = 0
        for local_index, point in enumerate(waypoints):
            index = precompleted + local_index
            self.active_waypoint_id = index
            self.active_waypoint_pub.publish(Int32(data=index))
            with self.lock:
                current = self.pose
            next_point = waypoints[
                min(local_index + 1, len(waypoints) - 1)]
            yaw = (math.atan2(next_point[1] - point[1], next_point[0] - point[0])
                   if next_point != point else math.atan2(
                       point[1] - current[1], point[0] - current[0]))
            if semantic_deadline is not None and time.monotonic() >= semantic_deadline:
                return {"success": False,
                        "reason": ("room_budget_timeout" if progressive_timeout else
                                   "semantic_goal_timeout"),
                        "failed_waypoint": index,
                        "waypoints": original_waypoint_count,
                        "completed_waypoints": index,
                        "zero_command_stall_recovery_count": recoveries}
            waypoint_deadline = semantic_deadline
            if progressive_timeout:
                progress_deadline = time.monotonic() + progress_timeout
                waypoint_deadline = (min(semantic_deadline, progress_deadline)
                                     if semantic_deadline is not None else
                                     progress_deadline)
            source = str(goal.get("source", ""))
            allow_door_preempt = bool(
                self.room_scheduler.active_door is None and
                (source in ("fuel_frontier", "corridor_sweep",
                            "corridor_forward_no_frontier",
                            "corridor_monotonic_forward_probe",
                            "corridor_remembered_side_branch",
                            "corridor_side_coverage_debt",
                            "corridor_side_branch_deepen") or
                 (source == "stair_corridor_exit_handoff" and
                  self._terminal_return_room_supplement_active())))
            result = self._publish_waypoint(
                point, z, yaw, waypoint_deadline,
                allow_local_door_preempt=allow_door_preempt)
            recoveries += int(result.get("zero_command_stall_recovery_count", 0))
            if result.get("reason") == "path_preempted_for_local_door":
                return {
                    "success": True,
                    "reason": "path_preempted_for_local_door",
                    "waypoints": original_waypoint_count,
                    "completed_waypoints": index,
                    "local_door_candidate_id":
                        result.get("local_door_candidate_id"),
                    "zero_command_stall_recovery_count": recoveries,
                }
            if not result.get("success"):
                failure_reason = result.get("reason", "executor_failure")
                if (progressive_timeout and
                        failure_reason == "semantic_goal_timeout"):
                    failure_reason = ("room_budget_timeout"
                                      if semantic_deadline is not None and
                                      time.monotonic() >= semantic_deadline else
                                      "exit_progress_stalled")
                return {"success": False, "reason": failure_reason,
                        "failed_waypoint": index,
                        "waypoints": original_waypoint_count,
                        "completed_waypoints": index,
                        "zero_command_stall_recovery_count": recoveries,
                        "progressive_timeout": progressive_timeout,
                        "progress_timeout_sec": progress_timeout}
        return {"success": True, "reason": "goal_reached",
                "waypoints": original_waypoint_count,
                "completed_waypoints": original_waypoint_count,
                "skipped_reached_prefix": precompleted,
                "zero_command_stall_recovery_count": recoveries}

    def _scan_check(self, x, y, yaw):
        request = CheckTwinCylinderRequest()
        request.pose_x, request.pose_y = x, y
        request.pose_z = self.pose[2]
        request.yaw = yaw
        request.front_offset = self.scan_config.body_front_offset
        request.rear_offset = self.scan_config.body_rear_offset
        request.radius = (self.scan_config.body_collision_radius +
                          self.scan_config.body_safety_margin)
        request.min_height = (self.scan_config.body_min_height -
                              self.scan_config.body_safety_margin)
        request.max_height = (self.scan_config.body_max_height +
                              self.scan_config.body_safety_margin)
        request.clearance_search_radius = request.radius + 0.50
        try:
            response = self.scan_service(request)
        except rospy.ServiceException:
            return FootprintCheck(map_available=False, status="map_unavailable")
        clearance = float(response.minimum_obstacle_clearance)
        return FootprintCheck(
            bool(response.map_available), bool(response.occupied_collision),
            int(response.unknown_queries), int(response.occupied_queries),
            clearance if math.isfinite(clearance) else None,
            (response.front_center_x, response.front_center_y),
            (response.rear_center_x, response.rear_center_y), response.status)

    @staticmethod
    def _max_yaw_change(yaws):
        return max((abs(math.atan2(math.sin(b - a), math.cos(b - a)))
                    for a, b in zip(yaws[:-1], yaws[1:])), default=0.0)

    def _refine_path(self, cycle, path_result, goal):
        if goal.get("verified_corridor_3d_probe"):
            # Every 10 cm sample of this short centreline segment was already
            # accepted by the independent live 3-D twin-cylinder service.
            # Reapplying the stale 2-D A* projection here would recreate the
            # exact false rejection this bounded F2 recovery is for.
            path_result["refinement_status"] = \
                "dense_live_3d_corridor_probe_verified"
            path_result["refined_path"] = list(path_result.get("path") or [])
            path_result["refined_path_length"] = polyline_length(
                path_result["refined_path"])
            return path_result
        if goal.get("verified_trajectory_backtrack"):
            # These short waypoints are the reverse of poses physically
            # occupied moments ago. A colliding current footprint is exactly
            # why this recovery exists, so rechecking that start pose would
            # reject every possible escape before motion begins.
            path_result["refinement_status"] = \
                "recent_trajectory_backtrack_verified"
            path_result["refined_path"] = list(path_result.get("path") or [])
            path_result["refined_path_length"] = polyline_length(
                path_result["refined_path"])
            return path_result
        mandatory = [tuple(map(float, point[:2]))
                     for point in (goal.get("mandatory_portal_waypoints") or [])
                     if isinstance(point, (list, tuple)) and len(point) >= 2]
        source = str(goal.get("source", ""))
        preserve_astar_geometry = (
            bool(goal.get("_preserve_astar_geometry", False)) or
            source in (
                "corridor_side_coverage_debt",
                "corridor_side_branch_deepen",
                "corridor_remembered_side_branch",
                "side_region_corridor_return",
            ))
        if mandatory:
            # Portal paths can include a several-metre corridor approach when
            # doorway evidence matures late.  At the normal 0.3 m spacing the
            # executor stops at every tiny A* sample and can time out before it
            # reaches the door.  Each retained chord is audited below by the
            # 3-D footprint refiner; the explicit portal anchors are still
            # segmented and therefore cannot be removed.
            # Portal anchors are already retained explicitly below and every
            # resulting chord is SCAN-lite footprint checked.  A 0.9 m
            # sampling interval still turned a 3 m doorway transit into five
            # stop-and-go commands (opt30), consuming the far-room budget.
            # Use longer audited chords between the anchors so the gait keeps
            # momentum through clear door approaches without skipping any
            # mandatory portal geometry.
            portal_spacing = max(self.waypoint_spacing, 1.45)
            original = segmented_execution_waypoints(
                path_result["path"], mandatory, portal_spacing, 0.08)
        elif preserve_astar_geometry:
            original = execution_waypoints(path_result["path"], self.waypoint_spacing)
        else:
            # Most FUEL corridor goals are deliberately short and visible.
            # Validate their direct chord with SCAN-lite first, preserving the
            # former continuous one-goal execution instead of making the robot
            # brake at every conservative A* grid bend.  If that chord is not
            # safe, the replan branch below retries with full A* geometry.
            original = ([tuple(path_result["path"][-1])]
                        if path_result.get("path") else [])
        if not self.enable_scan_lite:
            path_result["execution_waypoints"] = original
            return path_result
        with self.lock:
            current = self.pose
            recent_trajectory = list(self.trajectory[-20:])
            registration_healthy = self.fastlio_registration_healthy
        refinement_input = list(original)
        current_point = (float(current[0]), float(current[1]))
        if (not refinement_input or
                math.hypot(refinement_input[0][0] - current_point[0],
                           refinement_input[0][1] - current_point[1]) > 0.02):
            refinement_input.insert(0, current_point)
        target = goal.get("position") or []
        final_yaw = float(target[3]) if len(target) > 3 else None
        if mandatory:
            refiner_config = replace(
                self.scan_config, enable_waypoint_simplification=False)
        elif source == "corridor_sweep":
            # If waypoint zero is occupied, relocating it is not a valid path
            # repair: it is the robot's live pose.  Fail after one footprint
            # query so the recent-trajectory corridor escape below can run in
            # milliseconds rather than after a 7--10 second raster search.
            refiner_config = replace(
                self.scan_config, repair_start_waypoint=False)
        else:
            refiner_config = self.scan_config
        # The rolling 2-D projection can briefly label the live odometry cell
        # as occupied although the independent 3-D footprint service says the
        # robot is clear.  Previously SCAN-lite then rejected every corridor
        # continuation at waypoint zero; three such *map-start* rejections
        # were counted as a mission failure and ended exploration before the
        # far doors were reached.  Override only that one, already verified
        # footprint query.  Every point ahead and every segment still uses the
        # normal collision service.
        start_override = None
        raw_start_footprint = None
        start_override_reason = None
        start_override_radius = 0.06
        if (not mandatory and source == "corridor_sweep"):
            footprint = self._scan_check(current_point[0], current_point[1],
                                         float(current[3]))
            raw_start_footprint = footprint
            if footprint.map_available and not footprint.occupied_collision:
                start_override = footprint
                start_override_reason = "independent_3d_footprint_clear"
            elif (footprint.map_available and
                  footprint.occupied_collision and
                  corridor_live_start_override_allowed(
                      current, recent_trajectory, self.corridor_anchor,
                      self.corridor_axis, registration_healthy)):
                # The robot's live, continuous corridor pose is stronger
                # evidence for waypoint zero than a self/clutter voxel in the
                # rolling map.  Override only queries within 6 cm of that
                # pose; the first point ahead and the full route remain under
                # ordinary twin-cylinder collision checking.
                start_override = FootprintCheck(
                    map_available=True, occupied_collision=False,
                    unknown_queries=footprint.unknown_queries,
                    occupied_queries=footprint.occupied_queries,
                    minimum_obstacle_clearance=None,
                    front_center=footprint.front_center,
                    rear_center=footprint.rear_center,
                    status="verified_live_corridor_pose_override")
                start_override_reason = \
                    "healthy_continuous_live_corridor_pose"
                # Clear exactly one already-overlapping body envelope.  A
                # self-return at the live centre also overlaps the 10 cm
                # segment sample immediately ahead; overriding only the
                # mathematical point zero would therefore still fail before
                # leaving the same voxel.  This radius is bounded by the A1
                # twin-cylinder extent and is never applied to room/door
                # paths or to an unhealthy/discontinuous pose.
                start_override_radius = min(
                    0.25,
                    self.scan_config.body_collision_radius +
                    self.scan_config.body_safety_margin +
                    max(abs(self.scan_config.body_front_offset),
                        abs(self.scan_config.body_rear_offset)))

        def scan_check(point_x, point_y, yaw):
            if (start_override is not None and
                    math.hypot(float(point_x) - current_point[0],
                               float(point_y) - current_point[1]) <=
                    start_override_radius):
                return start_override
            return self._scan_check(point_x, point_y, yaw)

        refiner = PathRefiner(refiner_config, scan_check)
        refined = refiner.refine(refinement_input, final_yaw)
        refined_execution = list(refined.path)
        if (refined_execution and
                math.hypot(refined_execution[0][0] - current_point[0],
                           refined_execution[0][1] - current_point[1]) <= 0.05):
            refined_execution.pop(0)
        continuous_door_center_transit = False
        # The door centre is a geometric portal constraint, not a place where
        # the gait should brake to zero.  Stopping on the wall plane caused the
        # A1 to stall at otherwise valid seed-77 doors.  For ENTRY/EXIT, remove
        # that single execution stop when the refiner verifies that the chord
        # from the preceding waypoint to the following inside waypoint is
        # footprint-safe.  The centre remains in refined.path and in the
        # mandatory audit, so planning still cannot shortcut across a wall.
        portal_role = goal.get("room_role")
        if (refined.success and portal_role in ("ENTRY", "EXIT") and
                len(mandatory) >= 2 and len(refined_execution) >= 3):
            door_center = mandatory[-2]
            center_index = next((index for index, point in
                                 enumerate(refined_execution)
                                 if math.hypot(point[0] - door_center[0],
                                               point[1] - door_center[1]) <= .12),
                                None)
            if (center_index is not None and 0 < center_index <
                    len(refined_execution) - 1 and
                    refiner._segment_safe(
                        refined_execution[center_index - 1],
                        refined_execution[center_index + 1])):
                refined_execution.pop(center_index)
                continuous_door_center_transit = True
        mandatory_preserved = all(any(
            math.hypot(point[0] - anchor[0], point[1] - anchor[1]) <= .12
            for point in refined.path) for anchor in mandatory)
        self.scan_duration_pub.publish(Float64(data=refined.duration_ms))
        record = {
            "timestamp": time.time(), "elapsed_time": self.elapsed(), "goal_id": cycle,
            "original_waypoint_count": len(original),
            "refined_waypoint_count": len(refined_execution),
            "removed_waypoint_count": max(0, len(original) - len(refined_execution)),
            "modified_waypoint_count": len(refined.repaired_waypoints),
            "original_path_length": polyline_length(refinement_input),
            "refined_path_length": polyline_length(refined.path),
            "original_max_yaw_change": self._max_yaw_change(
                path_yaws(refinement_input, final_yaw)),
            "refined_max_yaw_change": self._max_yaw_change(refined.yaws),
            "collision_checks": refined.collision_checks,
            "colliding_original_waypoints": refined.colliding_original_waypoints,
            "repaired_waypoints": refined.repaired_waypoints,
            "unknown_queries": refined.unknown_queries,
            "occupied_queries": refined.occupied_queries,
            "minimum_obstacle_clearance": refined.minimum_obstacle_clearance,
            "refinement_duration_ms": refined.duration_ms,
            "refinement_success": refined.success, "failure_reason": (
                None if refined.success else refined.reason), "fallback_used": False,
            "mandatory_portal_waypoints": [list(point) for point in mandatory],
            "mandatory_portal_waypoints_preserved": mandatory_preserved,
            "continuous_door_center_transit": continuous_door_center_transit,
            "corridor_start_footprint_override": start_override is not None,
            "corridor_start_footprint_override_reason":
                start_override_reason,
            "corridor_start_footprint_override_radius_m": round(
                float(start_override_radius), 3)
                if start_override is not None else 0.0,
            "corridor_start_raw_collision": bool(
                raw_start_footprint is not None and
                raw_start_footprint.occupied_collision),
        }
        self.scan_history.append(record)
        debug_path = os.path.join(self.output_dir, "scan_lite_waypoint_debug.jsonl")
        try:
            with open(debug_path, "a", encoding="utf-8") as stream:
                for item in refined.debug:
                    payload = dict(item, timestamp=time.time(), goal_id=cycle)
                    stream.write(json.dumps(payload, sort_keys=True) + "\n")
                stream.flush()
        except OSError as error:
            rospy.logerr("SCAN-lite debug logging failed: %s", error)
        if refined.success:
            if mandatory and not mandatory_preserved:
                # In a narrow doorway the raster repair may move a mandatory
                # anchor by a few centimetres even though every original A*
                # anchor remains 3-D footprint-safe.  For a *short EXIT*
                # this must not strand the robot and burn all retries.  Keep
                # the exception tightly bounded: no entries/frontiers, no
                # long return, and every original point must be map-available
                # and collision-free in the live SCAN-lite service.
                short_exit = (portal_role == "EXIT" and
                              polyline_length(original) <= 2.5)
                original_safe = short_exit and bool(original)
                if original_safe:
                    for point in original:
                        footprint = self._scan_check(
                            point[0], point[1], float(current[3]))
                        if (not footprint.map_available or
                                footprint.occupied_collision):
                            original_safe = False
                            break
                if original_safe:
                    record["fallback_used"] = True
                    record["fallback_reason"] = (
                        "short_exit_original_anchors_after_refiner_anchor_loss")
                    path_result["execution_waypoints"] = original
                    path_result["refinement_status"] = (
                        "short_exit_original_anchor_fallback")
                    path_result["scan_lite_fallback_used"] = True
                    return path_result
                path_result.update({"success": False,
                                    "reason": "mandatory_portal_waypoint_lost",
                                    "execution_waypoints": []})
                return path_result
            path_result["execution_waypoints"] = refined_execution
            path_result["refined_path"] = refined.path
            path_result["refined_waypoints"] = refined_execution
            path_result["refinement_status"] = refined.reason
            path_result["refined_path_length"] = polyline_length(refined.path)
            return path_result
        # An EXIT route is first constructed with mandatory door anchors and
        # explicitly preflighted by A*.  SCAN-lite may still report a
        # disconnected *refined* segment when its raster repair clips a
        # narrow portal corner.  Do not strand the robot in a room after that
        # stronger preflight has succeeded: execute the original, already
        # verified anchor path.  This exception is intentionally EXIT-only;
        # entry and ordinary frontier paths retain the normal SCAN rejection.
        if (portal_role == "EXIT" and mandatory and
                bool(goal.get("portal_preflight_verified", False))):
            record["fallback_used"] = True
            record["fallback_reason"] = "exit_preflighted_astar_after_scan_disconnect"
            path_result["execution_waypoints"] = original
            path_result["refinement_status"] = "exit_preflighted_astar_fallback"
            path_result["scan_lite_fallback_used"] = True
            return path_result
        # A post-entry raster can fail to reconnect a narrow doorway even when
        # the live 3-D refiner found no collision on any original waypoint.
        # run90 rejected the same 2.27 m Room-3 exit twice for precisely this
        # reason and exhausted the mission's bounded room-exit retries. Permit
        # only a short EXIT whose complete original trace is available in the
        # live map and collision-free; entries, long routes and genuinely
        # colliding portal paths remain rejected.
        short_live_3d_exit = bool(
            portal_role == "EXIT" and mandatory and
            refined.reason == "disconnected_refined_segment" and
            not refined.colliding_original_waypoints and original and
            polyline_length(original) <= 2.5)
        if short_live_3d_exit:
            for point in original:
                footprint = self._scan_check(
                    point[0], point[1], float(current[3]))
                if (not footprint.map_available or
                        footprint.occupied_collision):
                    short_live_3d_exit = False
                    break
        if short_live_3d_exit:
            record["fallback_used"] = True
            record["fallback_reason"] = \
                "short_exit_live_3d_after_refiner_disconnect"
            path_result["execution_waypoints"] = original
            path_result["refinement_status"] = \
                "short_exit_live_3d_fallback"
            path_result["scan_lite_fallback_used"] = True
            return path_result
        staging_entry_fallback = bool(
            portal_role == "ENTRY" and
            goal.get("entry_staging_preflight_verified", False))
        if ((source in ("corridor_sweep", "lightweight_room_semantic") or
             staging_entry_fallback) and
                refined.reason == "disconnected_refined_segment" and
                not refined.colliding_original_waypoints):
            # The A* corridor route already has normal footprint clearance;
            # this failure is produced by raster waypoint repair disconnecting
            # two otherwise collision-free samples.  Retain the conservative
            # A* geometry instead of counting the same forward transit as a
            # repeated goal failure and abandoning exploration.  The same
            # conservative exception is used for room semantic viewpoints and
            # the bounded shallow ENTRY retry; both paths have already passed
            # the normal footprint/A* clearance check.
            path_result["execution_waypoints"] = execution_waypoints(
                path_result.get("path") or [], max(self.waypoint_spacing, 1.0))
            path_result["refinement_status"] = (
                "entry_staging_astar_after_refiner_disconnect"
                if staging_entry_fallback else
                "corridor_astar_after_refiner_disconnect")
            path_result["scan_lite_fallback_used"] = True
            return path_result
        if self.scan_failure_policy == "fallback_original_path":
            record["fallback_used"] = True
            path_result["execution_waypoints"] = original
            return path_result
        # A corridor path can be safe up to a newly observed obstacle and
        # become unrepairable only at a later waypoint.  Preserve that
        # checked prefix so the main loop can advance to it and replan from a
        # fresh local map instead of retrying the identical distant target.
        # Portal paths never use this: their mandatory anchors must either be
        # fully verified or rejected as a whole.
        safe_prefix = []
        if (not mandatory and source == "corridor_sweep" and
                refined.colliding_original_waypoints):
            first_collision = min(refined.colliding_original_waypoints)
            if first_collision >= 2:
                safe_prefix = list(refinement_input[1:first_collision])
        return {"success": False, "reason": "scan_lite_refinement_failed",
                "scan_lite_reason": refined.reason, "path": path_result["path"],
                "scan_lite_start_collision":
                    0 in refined.colliding_original_waypoints,
                "scan_lite_colliding_original_waypoints":
                    list(refined.colliding_original_waypoints),
                "scan_lite_safe_prefix": safe_prefix,
                "refinement_status": "scan_lite_refinement_failed",
                "request_astar_replan": self.scan_failure_policy == "request_astar_replan"}

    def _trajectory_length_from_jsonl(self, goal_id):
        """Integrate the persisted FAST-LIO samples for one Goal interval."""
        path = os.path.join(self.output_dir, "trajectory_timeseries.jsonl")
        records = []
        try:
            with open(path, encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if int(record.get("active_goal_id", -1)) == int(goal_id):
                        records.append(record)
        except (OSError, TypeError, ValueError) as error:
            rospy.logwarn("Goal %s trajectory JSONL unavailable: %s", goal_id, error)
            return None
        return integrate_trajectory(records) if len(records) >= 2 else None

    def _trajectory_metrics(self):
        with self.lock:
            trajectory = list(self.trajectory)
        length = sum(math.hypot(b[1] - a[1], b[2] - a[2])
                     for a, b in zip(trajectory[:-1], trajectory[1:]))
        displacement = (math.hypot(trajectory[-1][1] - trajectory[0][1],
                                   trajectory[-1][2] - trajectory[0][2])
                        if len(trajectory) >= 2 else 0.0)
        return length, displacement

    def _capture_room_exit_map_snapshot(self, room_id):
        """Persist the online 2-D map at a completed room exit.

        The final OctoMap can be remapped after a later FAST-LIO correction.
        Saving the exact grid that governed the completed exit makes room
        evidence inspectable without treating a later map as historical fact.
        """
        with self.lock:
            grid = self.grid
            trajectory = list(self.trajectory)
        if grid is None:
            return None
        index = len(self.room_exit_map_snapshots) + 1
        directory = os.path.join(self.output_dir, "room_exit_map_snapshots")
        os.makedirs(directory, exist_ok=True)
        filename = "room_exit_{:02d}_grid.npz".format(index)
        path = os.path.join(directory, filename)
        try:
            xy = np.asarray([(item[1], item[2]) for item in trajectory],
                            dtype=np.float32)
            np.savez_compressed(
                path, occupancy=np.asarray(grid.data, dtype=np.int16),
                resolution=float(grid.resolution), origin_x=float(grid.origin_x),
                origin_y=float(grid.origin_y), trajectory_xy=xy)
        except (OSError, ValueError) as error:
            rospy.logwarn("Could not save room-exit map snapshot: %s", error)
            return None
        record = {
            "room_id": room_id,
            "elapsed_sec": round(self.elapsed(), 3),
            "snapshot_file": os.path.join("room_exit_map_snapshots", filename),
            "trajectory_sample_count": int(len(trajectory)),
            "grid_shape": [int(grid.data.shape[0]), int(grid.data.shape[1])],
        }
        self.room_exit_map_snapshots.append(record)
        self._atomic_json(os.path.join(
            self.output_dir, "room_exit_map_snapshots.json"),
            self.room_exit_map_snapshots)
        return record

    def _write_all_logs(self):
        length, displacement = self._trajectory_metrics()
        mission_elapsed = self.elapsed() if self.mission_clock_started else None
        total_wall_elapsed = time.monotonic() - self.process_started
        corridor_entry_elapsed = (
            float(self.corridor_entry_elapsed_sec)
            if self.corridor_entry_elapsed_sec is not None else None)
        corridor_return_elapsed = (
            float(self.corridor_return_elapsed_sec)
            if self.corridor_return_elapsed_sec is not None else None)

        def elapsed_delta(end, start):
            if end is None or start is None or float(end) < float(start):
                return None
            return round(float(end) - float(start), 3)

        fourth_exit_elapsed = (
            float(self.room_exit_map_snapshots[self.room_target_count - 1][
                "elapsed_sec"])
            if len(self.room_exit_map_snapshots) >= self.room_target_count
            else None)
        entry_to_fourth_exit = elapsed_delta(
            fourth_exit_elapsed, corridor_entry_elapsed)
        fourth_exit_to_return = elapsed_delta(
            corridor_return_elapsed, fourth_exit_elapsed)
        entry_to_return = elapsed_delta(
            corridor_return_elapsed, corridor_entry_elapsed)
        corridor_full_loop_complete = bool(
            self._floor_handoff_room_requirement_met() and
            entry_to_fourth_exit is not None and
            fourth_exit_to_return is not None and
            entry_to_return is not None)
        self._atomic_json(os.path.join(self.output_dir, "startup_status.json"), {
            "task_mode": self.task_mode, "state": self.state,
            "advanced_features": {name: False for name in DISABLED_FEATURES},
            "lightweight_room_scheduler": self.room_scheduler.config.enabled,
            "robot_spawn_pose": self.spawn_pose,
            "startup_elapsed_sec": self.startup_elapsed_sec,
            "mission_budget_starts_after_startup": True,
            "mission_elapsed_sec": (round(mission_elapsed, 3)
                                    if mission_elapsed is not None else None),
            "total_wall_elapsed_sec": round(total_wall_elapsed, 3),
            # Kept for older readers; after MISSION_CLOCK_START it means only
            # mission elapsed time, never startup plus mission time.
            "elapsed_sec": (round(mission_elapsed, 3)
                            if mission_elapsed is not None else None),
            "exploration_end_elapsed_sec": self.exploration_end_elapsed,
            "room_exit_map_snapshot_count": len(self.room_exit_map_snapshots),
        })
        self._atomic_json(os.path.join(self.output_dir, "slam_status.json"), {
            "ready": self.first_odom_elapsed is not None and self.first_cloud_elapsed is not None,
            "odometry_topic": self.odom_topic, "registered_cloud_topic": self.cloud_topic,
            "first_odometry_elapsed_sec": self.first_odom_elapsed,
            "first_registered_cloud_elapsed_sec": self.first_cloud_elapsed,
        })
        self._atomic_json(os.path.join(
            self.output_dir, "exploration_goal_history.json"), self.goal_history)
        self._atomic_json(os.path.join(
            self.output_dir, "path_planning_history.json"), self.path_history)
        if self.enable_scan_lite:
            self._atomic_json(os.path.join(
                self.output_dir, "scan_lite_refinement_history.json"), self.scan_history)
        self._atomic_json(os.path.join(
            self.output_dir, "depth_breadth_history.json"), self.depth_breadth_history)
        self._atomic_json(os.path.join(
            self.output_dir, "corridor_sweep_history.json"),
            self.corridor_sweep_history)
        self._atomic_json(os.path.join(
            self.output_dir, "corridor_branch_status.json"),
            self.branch_scheduler.snapshot())
        self._atomic_json(os.path.join(
            self.output_dir, "visited_region_memory.json"),
            self.visited_region_samples)
        self._atomic_json(os.path.join(
            self.output_dir, "goal_execution_metrics.json"), self.goal_metrics)
        self._atomic_json(os.path.join(
            self.output_dir, "room_recognition_history.json"),
            self.room_scheduler.snapshot())
        self._atomic_json(os.path.join(self.output_dir, "baseline_summary.json"), {
            "task_mode": self.task_mode,
            "startup_elapsed_sec": self.startup_elapsed_sec,
            "mission_budget_starts_after_startup": True,
            "room_exit_map_snapshots": list(self.room_exit_map_snapshots),
            "fast_lio_ready_time_sec": self.first_odom_elapsed,
            "voxel_map_ready_time_sec": self.map_ready_elapsed,
            "first_fuel_goal_time_sec": self.first_goal_elapsed,
            "fuel_planning_count": self.fuel_plan_count,
            "astar_success_count": self.astar_success_count,
            "goal_execution_success_count": self.goal_success_count,
            "goal_execution_failure_count": self.goal_failure_count,
            "depth_breadth_scheduler_enabled": self.depth_breadth.config.enabled,
            "depth_breadth_region_count": self.depth_breadth.region_index,
            "depth_breadth_phase": self.depth_breadth.phase(self.elapsed()),
            "region_coverage_status": self.depth_breadth.coverage_status,
            "region_entry_return_pending":
                self.depth_breadth.entry_return_pending,
            "post_entry_lidar_observed_cell_count":
                len(self.post_entry_observed_cells),
            "corridor_branch_scheduler_enabled":
                self.enable_corridor_branch_scheduler,
            "corridor_branch_count": len(self.branch_scheduler.branches),
            "corridor_branch_merged_aliases":
                dict(self.branch_scheduler.aliases),
            "visited_region_sample_count":
                len(self.visited_region_samples),
            "corridor_branch_complete_count": sum(
                branch.state == "COVERED"
                for branch in self.branch_scheduler.branches),
            "corridor_branch_station_frame": {
                "origin": (
                    [float(value) for value in self.corridor_station_origin]
                    if self.corridor_station_origin is not None else None),
                "axis": (
                    [float(value) for value in self.corridor_station_axis]
                    if self.corridor_station_axis is not None else None),
            },
            "corridor_branches": self.branch_scheduler.snapshot(),
            "lightweight_room_scheduler_enabled": self.room_scheduler.config.enabled,
            "local_doorway_detector_enabled": self.enable_local_doorway_detector,
            "local_door_candidate_last_elapsed_sec": (
                self.last_local_door_received
                if math.isfinite(self.last_local_door_received) else None),
            "room_entry_candidate_last_elapsed_sec": (
                self.last_local_entry_received
                if math.isfinite(self.last_local_entry_received) else None),
            "local_rescan_count": self.local_rescan_count,
            "local_rescan_last_completed_elapsed_sec": (
                self.last_local_rescan_completed
                if math.isfinite(self.last_local_rescan_completed) else None),
            "frontier_termination_decision_count":
                len(self.termination_decisions),
            "recognized_room_count": self.room_scheduler.room_count,
            "exited_room_count": sum(
                door.visited for door in self.room_scheduler.detector.doors),
            "complete_room_count": sum(
                door.completed for door in self.room_scheduler.detector.doors),
            "room_target_count": self.room_target_count,
            "room_phase_target_seconds": self.room_phase_target_seconds,
            "room_phase_started_at_sec":
                self.room_scheduler.first_room_started_at,
            "fourth_room_exit_elapsed_sec": (
                self.room_exit_map_snapshots[self.room_target_count - 1][
                    "elapsed_sec"]
                if len(self.room_exit_map_snapshots) >= self.room_target_count
                else None),
            "fourth_room_exit_room_phase_elapsed_sec": (
                float(self.room_exit_map_snapshots[
                    self.room_target_count - 1]["elapsed_sec"]) -
                float(self.room_scheduler.first_room_started_at)
                if (len(self.room_exit_map_snapshots) >= self.room_target_count and
                    self.room_scheduler.first_room_started_at is not None)
                else None),
            "room_phase_target_met": bool(
                len(self.room_exit_map_snapshots) >= self.room_target_count and
                self.room_scheduler.first_room_started_at is not None and
                float(self.room_exit_map_snapshots[
                    self.room_target_count - 1]["elapsed_sec"]) -
                float(self.room_scheduler.first_room_started_at) <=
                self.room_phase_target_seconds),
            "corridor_entry_elapsed_sec": corridor_entry_elapsed,
            "corridor_entry_source": self.corridor_entry_source,
            "corridor_return_elapsed_sec": corridor_return_elapsed,
            "corridor_return_reason": self.corridor_return_reason,
            "corridor_return_pose": (
                list(self.corridor_return_pose)
                if self.corridor_return_pose is not None else None),
            "corridor_return_verified":
                self.corridor_return_elapsed_sec is not None,
            "corridor_entry_to_fourth_room_exit_elapsed_sec":
                entry_to_fourth_exit,
            "corridor_entry_to_fourth_room_exit_target_met": bool(
                entry_to_fourth_exit is not None and
                entry_to_fourth_exit <= self.room_phase_target_seconds),
            "fourth_room_exit_to_corridor_return_elapsed_sec":
                fourth_exit_to_return,
            "corridor_entry_to_return_elapsed_sec": entry_to_return,
            "corridor_full_loop_elapsed_sec": (
                entry_to_return if corridor_full_loop_complete else None),
            "corridor_full_loop_complete": corridor_full_loop_complete,
            "post_room_corridor_sweep": {
                "enabled": self.enable_post_room_corridor_sweep,
                "attempts": self.post_room_corridor_sweep_attempts,
                "max_goals": self.post_room_corridor_sweep_max_goals,
                "done": self.post_room_corridor_sweep_done,
            },
            "mission_start_pose": self.mission_start_pose,
            "stair_wait_anchor": self.stair_wait_anchor,
            "stair_wait_tolerance_m": self.stair_wait_tolerance,
            "mission_home_tolerance_m": self.mission_home_tolerance,
            "mission_home_reached": self._home_reached(),
            "corridor_sweep": {
                "established": self.corridor_established,
                "confirmation_count": self.corridor_confirmation_count,
                "confirmation_updates_required":
                    self.corridor_confirmation_updates,
                "minimum_visible_length_m":
                    self.corridor_minimum_visible_length,
                "minimum_aspect_ratio":
                    self.corridor_minimum_aspect_ratio,
                "minimum_advance_m": self.corridor_minimum_advance,
                "recent_exclusion_m": self.corridor_recent_exclusion,
                "forward_distance_m": self.corridor_forward_distance,
                "reversal_count": self.corridor_reversal_count,
                "reversal_limit": self.corridor_reversal_limit,
                "anchor": ([float(self.corridor_anchor[0]),
                            float(self.corridor_anchor[1])]
                           if self.corridor_anchor is not None else None),
                "direction_aligned": self.corridor_direction_aligned,
            },
            "corridor_monotonic_forward_probe": {
                "enabled": self.enable_corridor_monotonic_forward_probe,
                "selected_count": self.corridor_monotonic_probe_count,
                "mode_active": self.corridor_monotonic_probe_mode_active,
                "activation_after_failed_recoveries":
                    self.corridor_monotonic_probe_after,
                "probe_distance_m": self.corridor_monotonic_probe_distance,
                "probe_limit": self.corridor_monotonic_probe_limit,
            },
            "total_trajectory_length_m": round(length, 4),
            "net_displacement_m": round(displacement, 4),
            "termination_reason": self.termination_reason or "running",
        })
        with self.lock:
            trajectory = list(self.trajectory)
        self._atomic_json(os.path.join(self.output_dir, "trajectory_path.json"), {
            "schema": "simenv_fastlio_trajectory_path_v1",
            "source": "FAST-LIO nav_msgs/Odometry",
            "sample_rate_limit_hz": 10.0,
            "samples": [{"elapsed_time": item[0], "position_x": item[1],
                         "position_y": item[2], "position_z": item[3],
                         "yaw": item[4], "frame_id": item[5]}
                        for item in trajectory],
        })

    def _generate_visualization(self):
        if not self.auto_generate_visualization or self.visualization_generated:
            return
        self.visualization_generated = True
        with self.lock:
            before = self.map_saved_count
        self.finalize_map_pub.publish(Bool(data=True))
        deadline = time.monotonic() + 5.0
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            with self.lock:
                if self.map_saved_count > before:
                    break
            time.sleep(0.05)
        if not os.path.isfile(self.visualization_script):
            rospy.logerr("Visualization script missing: %s", self.visualization_script)
            return
        command = [sys.executable, self.visualization_script,
                   "--run-dir", self.output_dir,
                   "--output-dir", os.path.join(self.output_dir, "visualization")]
        try:
            completed = subprocess.run(command, timeout=120.0, check=False)
            if completed.returncode:
                rospy.logerr("Offline visualization exited with code %d",
                             completed.returncode)
            else:
                rospy.loginfo("Offline visualization saved under %s/visualization",
                              self.output_dir)
        except (OSError, subprocess.TimeoutExpired) as error:
            rospy.logerr("Offline visualization failed: %s", error)

    def _sync_localization_recovery_diagnostics_locked(self):
        window = self._localization_recovery_window
        self._localization_recovery_active = bool(window.active)
        self._localization_recovery_started_at = window.started_at
        self._localization_recovery_attempts = int(window.attempts)

    def _publish_truth_recovery_request(self, reason, invalid_count):
        payload = {
            "reason": str(reason),
            "invalid_count": int(invalid_count),
            "elapsed_sec": round(self.elapsed(), 3),
            "attempt": int(self._localization_recovery_attempts),
        }
        self._truth_recovery_request_pub.publish(String(data=json.dumps(payload)))

    def _confirmed_local_door_ahead(self, pose, grid, axis,
                                    lookahead=None):
        """True when the local detector has a fresh confirmed doorway ahead.

        A confirmed aperture between the robot and the nominal terminal-wall
        scan range proves the corridor has not ended.  Without this guard the
        5 m forward-wall scan treats the wall band immediately behind a real
        doorway pair as a corridor terminus (f2_f3_fix_2 returned at y=26.5
        and never reached the row-2 doorway at y=28.6).

        The detector's confirmed stream is latched, so freshness is checked
        explicitly: only evidence published within the last few seconds is
        trusted to veto a terminal return.
        """
        if (not self.enable_local_doorway_detector or pose is None or
                axis is None or grid is None):
            return False
        with self.lock:
            status = dict(self.local_entry_status or {})
            received = self.last_local_entry_received
        if (not status or
                self.elapsed() - received >
                max(3.0, float(self.local_candidate_freshness))):
            return False
        direction = np.asarray(axis, dtype=float)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            return False
        direction /= norm
        limit = (float(self.corridor_terminal_return_distance)
                 if lookahead is None else float(lookahead))
        for candidate in status.get("candidates") or []:
            if not candidate.get("confirmed"):
                continue
            point = candidate.get("door_center") or []
            if len(point) < 2:
                continue
            delta = (float(point[0]) - float(pose[0])) * direction[0] + \
                    (float(point[1]) - float(pose[1])) * direction[1]
            if 0.0 < delta <= limit:
                return True
        return False

    def _floor_handoff_room_requirement_met(self):
        """Apply the optional strict all-physical-EXIT floor gate."""
        if not getattr(self, "require_all_rooms_for_floor_handoff", True):
            return True
        # A production three-floor mission always owns four rooms.  Protect
        # the handoff even if a launch accidentally lowers room_target_count;
        # a larger configured target remains authoritative.
        required_exits = max(4, int(self.room_target_count))
        return bool(
            self.room_scheduler.active_door is None and
            sum(bool(door.visited) for door in
                self.room_scheduler.detector.doors) >= required_exits)

    def _handle_upper_floor_truth_probe_result(
            self, success, source, reported_reason, exited_rooms):
        """Keep a blocked upper-floor probe local and start a safe return.

        Two consecutive rejected centreline probes are sufficient evidence to
        stop outbound probing, even before the first room is committed.  The normal terminal
        return path remains door-preemptible, so a missed opposite room can be
        entered and immediately exited on the way back to the stair.
        """
        if str(source) != "upper_floor_truth_corridor_probe":
            return False
        if success:
            self.upper_floor_truth_probe_failure_count = 0
            return True
        count = int(getattr(
            self, "upper_floor_truth_probe_failure_count", 0)) + 1
        self.upper_floor_truth_probe_failure_count = count
        history = getattr(self, "corridor_sweep_history", None)
        if isinstance(history, list):
            history.append({
                "event": "UPPER_FLOOR_TRUTH_PROBE_FAILURE_DEFERRED",
                "elapsed_sec": round(self.elapsed(), 3),
                "failure_count": count,
                "reason": str(reported_reason),
                "exited_rooms": int(exited_rooms),
            })
        # A failed projection-backed probe before any physical ROOM_EXIT is
        # not evidence of a terminal corridor: the freshly re-anchored upper
        # floor map may still contain only the landing slice.  Keep probing
        # and growing that map; a return at 0/4 would make the required
        # room contract impossible.
        if int(exited_rooms) == 0:
            if isinstance(history, list):
                history.append({
                    "event": "UPPER_FLOOR_ZERO_ROOM_PROBE_RETRY",
                    "elapsed_sec": round(self.elapsed(), 3),
                    "failure_count": count,
                    "policy": "preserve_forward_truth_probe_until_door_evidence",
                })
            return True
        latch_threshold = 2
        if count >= latch_threshold:
            self.corridor_terminal_return_latched = True
            self.corridor_partial_return_trigger_reason = (
                "blocked_upper_floor_corridor_probe")
            if isinstance(history, list):
                history.append({
                    "event": "UPPER_FLOOR_BLOCKED_PROBE_RETURN_LATCHED",
                    "elapsed_sec": round(self.elapsed(), 3),
                    "failure_count": count,
                    "policy": "return_with_missed_room_supplement",
                    "exited_rooms_at_latch": int(exited_rooms),
                })
        return True

    def _orient_corridor_axis_to_forward_hint(self, axis):
        """Choose the outbound sign of an otherwise symmetric fresh axis."""
        candidate = np.asarray(axis, dtype=float).copy()
        if candidate.shape != (2,) or not np.all(np.isfinite(candidate)):
            return candidate
        norm = float(np.linalg.norm(candidate))
        if norm < 1e-9:
            return candidate
        candidate /= norm
        hint = getattr(self, "corridor_forward_axis_hint", None)
        if (self.corridor_axis is None and hint is not None and
                float(np.dot(candidate, hint)) < 0.0):
            candidate = -candidate
        return candidate

    def _stable_corridor_axes(self, raw_axis):
        """Return an axis/normal pair with a persistent side convention.

        Local parallel-wall fits may flip their axis by pi between scans. A
        flipped axis also flips the normal and turns a real opposite doorway
        into a same-side/rearward candidate. Once the online corridor station
        frame exists, align every local fit to it before assigning door sides.
        """
        axis = np.asarray(raw_axis, dtype=float)
        axis /= max(1e-9, float(np.linalg.norm(axis)))
        reference = getattr(self, "corridor_station_axis", None)
        if reference is not None:
            reference = np.asarray(reference, dtype=float)
            reference /= max(1e-9, float(np.linalg.norm(reference)))
            if float(np.dot(axis, reference)) < 0.0:
                axis = -axis
        return axis, np.asarray([-axis[1], axis[0]], dtype=float)

    def _record_corridor_entry(self, anchor, source, elapsed_sec=None,
                               append_history=True):
        """Latch the physical corridor entrance and its mission-clock time."""
        if anchor is None or len(anchor) < 2:
            return False
        try:
            z = (float(anchor[2]) if len(anchor) >= 3 else
                 float(self.pose[2]) if self.pose is not None else 0.0)
            normalized = (float(anchor[0]), float(anchor[1]), z)
            recorded_elapsed = round(
                self.elapsed() if elapsed_sec is None else float(elapsed_sec),
                3)
        except (TypeError, ValueError, IndexError):
            return False
        # The first corridor-ingress proof owns the timing origin.  It also
        # deliberately replaces the older "first successful motion" stair
        # anchor, which could lie in the lobby rather than at the corridor
        # mouth and made the measured loop endpoint ambiguous.
        if self.corridor_entry_elapsed_sec is not None:
            return False
        self.corridor_entry_anchor = normalized
        self.stair_wait_anchor = normalized
        self.corridor_entry_elapsed_sec = recorded_elapsed
        self.corridor_entry_source = str(source)
        if append_history:
            self.corridor_sweep_history.append({
                "event": "CORRIDOR_ENTRY_ANCHOR_RECORDED",
                "elapsed_sec": recorded_elapsed,
                "anchor": list(normalized[:2]),
                "source": self.corridor_entry_source,
                "stair_wait_anchor_synchronized": True,
            })
        return True

    def _record_corridor_return(self, reason, pose=None):
        """Record the first verified return to the corridor entrance."""
        if self.corridor_return_elapsed_sec is not None:
            return False
        with self.lock:
            current = pose if pose is not None else self.pose
        target = self._stair_wait_target()
        if current is None or target is None:
            return False
        try:
            normalized = (float(current[0]), float(current[1]),
                          float(current[2]) if len(current) >= 3 else 0.0)
            anchor_error = math.hypot(
                normalized[0] - float(target[0]),
                normalized[1] - float(target[1]))
        except (TypeError, ValueError, IndexError):
            return False
        self.corridor_return_elapsed_sec = round(self.elapsed(), 3)
        self.corridor_return_reason = str(reason)
        self.corridor_return_pose = normalized
        self.corridor_sweep_history.append({
            "event": "CORRIDOR_RETURN_RECORDED",
            "elapsed_sec": self.corridor_return_elapsed_sec,
            "entry_anchor": list(target[:2]),
            "return_pose": list(normalized[:2]),
            "anchor_error_m": round(anchor_error, 3),
            "reason": self.corridor_return_reason,
            "all_required_rooms_physically_exited":
                self._mission_rooms_physically_exited(),
        })
        return True

    def _recent_exit_needs_opposite_check(self, pose):
        # True only while the just-exited corridor station lacks its pair.
        exit_side = self._recent_room_exit_opposite_side(pose)
        if exit_side is None or self.last_room_exit_pose is None:
            return False
        # A mirrored or measured opposite ENTRY already moved the body inside
        # that room.  Holding the corridor station then blocks in-room G1/G3
        # planning for the remainder of the floor budget (speed_fix_5: room_02
        # spent 267 s spinning in G1_CENTER after a successful opposite ENTRY).
        if (self.room_scheduler.active_door is not None and
                self.room_scheduler.entry_confirmed):
            return False
        exit_station = self._corridor_station(self.last_room_exit_pose)
        paired = any(
            branch.state == "COVERED" and
            int(branch.side) == -int(exit_side) and
            abs(float(branch.station) - float(exit_station)) <=
            self.corridor_branch_station_tolerance
            for branch in self.branch_scheduler.branches)
        # Branch stations can retain their pre-rebase longitudinal value.
        # Reuse the same stronger physical evidence as
        # _recent_room_exit_opposite_side: a completed measured door on the
        # opposite side at this portal station means the pair is finished.
        # run corridor_rooms_140s_2 otherwise spent 47 s rechecking room 0/1.
        if (not paired and
                getattr(self, "corridor_station_axis", None) is not None and
                getattr(self, "corridor_station_origin", None) is not None):
            axis = np.asarray(self.corridor_station_axis, dtype=float)
            axis /= max(1e-9, float(np.linalg.norm(axis)))
            normal = np.asarray([-axis[1], axis[0]], dtype=float)
            origin = np.asarray(self.corridor_station_origin, dtype=float)
            station_tolerance = max(
                2.50, float(self.corridor_branch_station_tolerance))
            scheduler = getattr(self, "room_scheduler", None)
            detector = getattr(scheduler, "detector", None)
            for door in getattr(detector, "doors", []):
                if (not bool(getattr(door, "visited", False)) or
                        not bool(getattr(door, "completed", False))):
                    continue
                door_point = np.asarray(
                    getattr(door, "interior_side", None) or door.center,
                    dtype=float)
                door_side = (1 if float(np.dot(
                    door_point[:2] - origin, normal)) >= 0.0 else -1)
                if (door_side == -int(exit_side) and
                        abs(self._corridor_station(door.center) -
                            exit_station) <= station_tolerance):
                    paired = True
                    break
        return not paired

    def _plan_mirrored_paired_room_entry_goal(self,
                                               allow_recent_exit=False):
        """Recover one missed opposite room from a verified visited doorway.

        Ordinarily this is a terminal-return supplement.  Immediately after a
        physical room exit it may also enforce the same-station opposite-room
        check before forward motion.  A mirrored hypothesis still must pass
        the ordinary observed occupancy, A* and portal validation gates.
        """
        exited = sum(bool(door.visited) for door in
                     self.room_scheduler.detector.doors)
        with self.lock:
            pose, grid = self.pose, self.grid
        recent_exit_check = bool(
            allow_recent_exit and pose is not None and
            self._recent_room_exit_opposite_side(pose) is not None)
        if (not (self._terminal_return_room_supplement_active() or
                 recent_exit_check) or
                not (1 <= exited < self.room_target_count) or
                self.room_scheduler.active_door is not None or
                self.corridor_station_origin is None):
            return None
        if pose is None or grid is None:
            return None
        axis_source = (self.corridor_station_axis
                       if self.corridor_station_axis is not None else
                       self.corridor_axis)
        if axis_source is None:
            return None
        axis = np.asarray(axis_source, dtype=float)
        axis /= max(1e-9, float(np.linalg.norm(axis)))
        origin = np.asarray(self.corridor_station_origin, dtype=float)
        # The station origin is captured while entering the corridor and can
        # retain a sizeable lateral offset (for example after leaving the
        # first room).  Mirroring about that biased line makes a real room
        # look too shallow and silently drops its opposite-side partner.  A
        # visited doorway gives us a stronger online centreline observation:
        # its corridor_side point is the verified point to which the exit
        # returned.  Re-centre only the normal component using the median of
        # those physical exits; keep the longitudinal station frame intact.
        normal = np.asarray([-axis[1], axis[0]], dtype=float)
        corridor_side_offsets = []
        for door in self.room_scheduler.detector.doors:
            if not bool(getattr(door, "visited", False)):
                continue
            corridor_side = getattr(door, "corridor_side", None)
            if corridor_side is None:
                continue
            corridor_side_offsets.append(float(np.dot(
                np.asarray(corridor_side, dtype=float) - origin, normal)))
        if corridor_side_offsets:
            lateral_recenter = float(np.median(corridor_side_offsets))
            origin = origin + normal * lateral_recenter
        candidates = []
        for branch in self.branch_scheduler.branches:
            if branch.state == "UNENTERED":
                continue
            paired = any(
                other.branch_id != branch.branch_id and
                other.state == "COVERED" and
                int(other.side) == -int(branch.side) and
                abs(float(other.station) - float(branch.station)) <=
                self.corridor_branch_station_tolerance and
                abs(float(np.dot(
                    np.asarray(other.entry_target, dtype=float) - origin,
                    normal))) >= 0.90
                for other in self.branch_scheduler.branches)
            if paired:
                continue
            entry = np.asarray(branch.entry_target, dtype=float)
            centerline = origin + axis * float(np.dot(entry - origin, axis))
            mirrored = 2.0 * centerline - entry
            lateral = mirrored - centerline
            depth = float(np.linalg.norm(lateral))
            # A physically completed room is stronger evidence than the
            # shallow branch target used while approaching its doorway.  On
            # F3 the far narrow portal produced a valid COVERED branch only
            # 0.813 m from the rebased centreline; the former universal
            # 0.90 m gate silently discarded its still-missing opposite room.
            # Relax this gate only when a completed measured doorway matches
            # the branch's side and station.  The mirrored portal must still
            # pass the full observed-free ray, A* and SCAN-lite checks below.
            completed_source_door = any(
                bool(getattr(door, "visited", False)) and
                bool(getattr(door, "completed", False)) and
                (1 if float(np.dot(
                    np.asarray(getattr(door, "interior_side", None) or
                               door.center, dtype=float) - origin,
                    normal)) >= 0.0 else -1) == int(branch.side) and
                abs(self._corridor_station(door.center) -
                    self._corridor_station(branch.entry_target)) <=
                max(2.50, float(self.corridor_branch_station_tolerance))
                for door in self.room_scheduler.detector.doors)
            minimum_source_depth = 0.70 if completed_source_door else 0.90
            if depth < minimum_source_depth:
                continue
            station_delta = np.asarray(pose[:2], dtype=float) - centerline
            longitudinal_station_error = abs(float(np.dot(
                station_delta, axis)))
            lateral_centerline_error = abs(float(np.dot(
                station_delta, normal)))
            # Immediately after ROOM_EXIT the robot is already at the same
            # corridor *row* as the opposite door.  Recentring laterally to a
            # historical corridor-side reference moves it away from that
            # door (run31: 0.76 m and 0.62 m, about four seconds total) and
            # adds no portal evidence.  In this tightly gated recent-exit
            # case align only the longitudinal station. Terminal-return
            # recovery retains the conservative Euclidean recenter check.
            distance_to_station = (
                longitudinal_station_error if recent_exit_check else
                float(np.linalg.norm(station_delta)))
            if distance_to_station > 4.0:
                continue
            key = "mirrored-pair-{}".format(branch.branch_id)
            if self.elapsed() - self.local_candidate_attempts.get(
                    key, -math.inf) < 45.0:
                continue
            # A failed station recenter (goal_footprint_blocked) must not be
            # re-selected every replan cycle.  Reuse the cooldown recorded by
            # the goal-result handler so the same inferred pair waits before
            # being re-tested from its corridor station.
            recenter_key = "mirrored-recenter-{:.1f}-{:.1f}".format(
                float(centerline[0]), float(centerline[1]))
            recenter_cooldowns = getattr(
                self, "mirrored_pair_recenter_cooldowns", {})
            if self.elapsed() - recenter_cooldowns.get(
                    recenter_key, -math.inf) < 45.0:
                continue
            direction = lateral / depth
            # A branch entry target is an interior point, not the physical
            # door plane.  Sparse rolling-map evidence can make it only
            # ~1.25 m from a biased corridor line.  Multiplying that depth by
            # 0.65 placed run32's inferred far-row door only 1.06 m from the
            # already visited opposite portal; the duplicate-door guard then
            # correctly rejected it and the scheduler spent 13.9 s rescanning
            # a pair whose existence was already established.  Keep the
            # inferred plane near the observed corridor wall (0.90--1.10 m
            # from the centre estimate).  The complete shifted normal ray is
            # still required to be observed-free below before any ENTRY is
            # promoted.
            door_plane_depth = min(1.10, max(0.90, 0.85 * depth))
            door_center = centerline + direction * door_plane_depth
            interior = centerline + direction * max(1.75, depth)
            candidates.append((distance_to_station, branch, key, centerline,
                               door_center, interior))
        if not candidates:
            return None
        distance_to_station, branch, key, centerline, door_center, interior = min(
            candidates, key=lambda item: item[0])
        # Observe the inferred pair from its exact corridor station before
        # attempting a portal crossing.  run220 returned several metres away,
        # so the wall aperture was projected with the wrong normal and the
        # safe passage-ray check correctly rejected it.
        if distance_to_station > max(self.reached_tolerance, 0.45):
            self.corridor_sweep_history.append({
                "event": "MIRRORED_PAIR_STATION_RECENTER_SELECTED",
                "elapsed_sec": round(self.elapsed(), 3),
                "source_branch_id": branch.branch_id,
                "source_station_m": branch.station,
                "target": [float(centerline[0]), float(centerline[1])],
                "distance_m": round(float(distance_to_station), 3),
                "longitudinal_station_error_m": round(
                    longitudinal_station_error, 3),
                "lateral_centerline_error_m": round(
                    lateral_centerline_error, 3),
                "policy": "corridor_only_observation_before_portal_validation",
            })
            return {
                "position": [float(centerline[0]), float(centerline[1]),
                             float(pose[2])],
                "information_gain": 0.0, "score": 0.0,
                "source": "mirrored_pair_station_recenter",
                "scheduler_phase": "MIRRORED_PAIR_STATION_RECENTER",
            }
        self.local_candidate_attempts[key] = self.elapsed()
        # Opposite doors are paired by corridor row, but the two physical
        # aperture centres need not land in the same 15 cm occupancy cell.
        # F2 speed_fix_11 had valid free rays at +0.15/+0.30 m while the exact
        # mirror crossed one smeared wall cell. Search only along the latched
        # corridor axis, retain the same side/depth, and require every sample
        # from corridor to interior to be observed free.
        original_centerline = np.asarray(centerline, dtype=float)
        original_door_center = np.asarray(door_center, dtype=float)
        original_interior = np.asarray(interior, dtype=float)
        portal_shifts = []
        ray_step = max(0.05, min(0.10, 0.50 * float(grid.resolution)))
        for station_offset in (0.0, 0.15, -0.15, 0.30, -0.30,
                               0.45, -0.45, 0.60, -0.60,
                               0.75, -0.75):
            delta = axis * float(station_offset)
            shifted_centerline = original_centerline + delta
            shifted_door_center = original_door_center + delta
            shifted_interior = original_interior + delta
            start = (float(shifted_centerline[0]),
                     float(shifted_centerline[1]))
            finish = (float(shifted_interior[0]),
                      float(shifted_interior[1]))
            if _segment_has_occupied(grid, start, finish):
                continue
            length = float(np.linalg.norm(shifted_interior -
                                          shifted_centerline))
            sample_count = max(2, int(math.ceil(length / ray_step)))
            samples = [
                (float(shifted_centerline[0] +
                       ratio * (shifted_interior[0] -
                                shifted_centerline[0])),
                 float(shifted_centerline[1] +
                       ratio * (shifted_interior[1] -
                                shifted_centerline[1])))
                for ratio in np.linspace(0.0, 1.0, sample_count + 1)]
            if any(_state(grid, point) != 0 for point in samples):
                continue
            minimum_ray_clearance = min(
                _clearance(grid, point) for point in samples)
            portal_shifts.append((-float(minimum_ray_clearance),
                                  abs(float(station_offset)),
                                  float(station_offset),
                                  shifted_centerline, shifted_door_center,
                                  shifted_interior))
        if not portal_shifts:
            # Immediately after exiting the first room at a corridor station,
            # its opposite aperture can still be unknown/occupied in the
            # rolling raster.  Passing that unverified mirror to the room
            # scheduler makes its independent ray gate quarantine the useful
            # candidate for 12 s; by then normal corridor motion has moved
            # away from the paired station.  Refresh exactly once in place,
            # then require this entire observed-free sampling loop to pass on
            # the next cycle.  No unverified portal is ever commanded.
            refresh_counts = getattr(
                self, "mirrored_pair_portal_refresh_counts", {})
            self.mirrored_pair_portal_refresh_counts = refresh_counts
            refresh_count = int(refresh_counts.get(key, 0))
            if refresh_count < 1:
                refresh_counts[key] = refresh_count + 1
                # Use the same settled 2 rad/s envelope as room RGB-D fans.
                # Back-date the normal 45 s attempt clock just enough for an
                # immediate post-scan retry.
                self.local_candidate_attempts[key] = self.elapsed() - 43.0
                self.corridor_sweep_history.append({
                    "event": "MIRRORED_PAIR_PORTAL_MAP_REFRESH_SELECTED",
                    "elapsed_sec": round(self.elapsed(), 3),
                    "source_branch_id": branch.branch_id,
                    "source_station_m": branch.station,
                    "refresh_attempt": refresh_count + 1,
                    "policy": "one_half_turn_then_repeat_full_free_ray_gate",
                })
                self._perform_local_rescan(
                    angle_rad=math.pi, angular_speed=self.room_camera_sweep_speed,
                    reason="mirrored_pair_portal_geometry_refresh",
                    visual_sweep=True, room_id=None)
            else:
                self.local_candidate_attempts[key] = self.elapsed()
                self.corridor_sweep_history.append({
                    "event": "MIRRORED_PAIR_PORTAL_STILL_BLOCKED_AFTER_REFRESH",
                    "elapsed_sec": round(self.elapsed(), 3),
                    "source_branch_id": branch.branch_id,
                    "source_station_m": branch.station,
                    "policy": "cooldown_and_continue_corridor_without_entry",
                })
            return None
        if portal_shifts:
            (_, _, station_offset, centerline, door_center, interior) = min(
                portal_shifts, key=lambda item: item[:2])
            if abs(station_offset) > 0.01:
                self.corridor_sweep_history.append({
                    "event": "MIRRORED_PAIR_TANGENTIAL_PORTAL_SHIFTED",
                    "elapsed_sec": round(self.elapsed(), 3),
                    "source_branch_id": branch.branch_id,
                    "station_offset_m": round(station_offset, 3),
                    "policy": "best_observed_free_normal_ray_near_pair_station",
                })
            refresh_counts = getattr(
                self, "mirrored_pair_portal_refresh_counts", None)
            if refresh_counts is not None:
                refresh_counts.pop(key, None)
        yaw = math.atan2(float(interior[1] - door_center[1]),
                         float(interior[0] - door_center[0]))
        evidence = {
            "candidate_id": key, "confirmed": True,
            "confirmation_count": 3, "astar_reachable": True,
            "scan_lite_safe": True, "width_m": 1.20, "yaw": yaw,
            "corridor_side": [float(centerline[0]), float(centerline[1])],
            "door_center": [float(door_center[0]), float(door_center[1])],
            "entry_goal": [float(interior[0]), float(interior[1])],
            "unknown_behind_m2": 0.0, "free_behind_m2": 0.0,
            "open_area_m2": 0.0, "side": -int(branch.side),
        }
        activated = self.room_scheduler.consider_local_door_candidate(
            grid, (float(pose[0]), float(pose[1])), evidence, self.elapsed(),
            reuse_prevalidated_entry=True)
        if activated is None:
            # The mirrored normal ray is already observed-free here. If the
            # scheduler still declines it while the just-exited body remains
            # off the corridor centreline, improve measured doorway geometry
            # with one short safe recenter before spending time rotating.
            # run93 ultimately found the real pair only after that recenter.
            if (recent_exit_check and
                    longitudinal_station_error <= max(
                        self.reached_tolerance, 0.45) and
                    lateral_centerline_error <= 0.80):
                # The robot is already at the paired longitudinal station and
                # no unverified portal is being commanded. A separate
                # 0.5--0.8 m centreline goal followed by the same half-turn
                # added one extra stop/start without improving the normal-ray
                # gate (direction1: 5.6 s combined). Refresh the measured
                # aperture in place; local-door detection must still confirm
                # it before ENTRY, so the safety contract is unchanged.
                self.local_candidate_attempts[key] = self.elapsed() - 40.0
                self.corridor_sweep_history.append({
                    "event": "MIRRORED_PAIR_IN_PLACE_REFRESH_SELECTED",
                    "elapsed_sec": round(self.elapsed(), 3),
                    "source_branch_id": branch.branch_id,
                    "source_station_m": branch.station,
                    "longitudinal_station_error_m": round(
                        longitudinal_station_error, 3),
                    "lateral_centerline_error_m": round(
                        lateral_centerline_error, 3),
                    "policy":
                        "same_station_refresh_without_redundant_recenter",
                })
                self._perform_local_rescan(
                    angle_rad=math.pi,
                    angular_speed=self.room_camera_sweep_speed,
                    reason="mirrored_pair_portal_geometry_refresh",
                    visual_sweep=True, room_id=None)
                return None
            if (recent_exit_check and
                    lateral_centerline_error > max(
                        self.reached_tolerance, 0.45)):
                self.local_candidate_attempts[key] = self.elapsed() - 45.0
                self.corridor_sweep_history.append({
                    "event": "MIRRORED_PAIR_REJECTED_RECENTER_SELECTED",
                    "elapsed_sec": round(self.elapsed(), 3),
                    "source_branch_id": branch.branch_id,
                    "source_station_m": branch.station,
                    "target": [float(centerline[0]),
                               float(centerline[1])],
                    "lateral_centerline_error_m": round(
                        lateral_centerline_error, 3),
                    "action": "safe_centerline_recenter_before_rescan",
                })
                return {
                    "position": [float(centerline[0]), float(centerline[1]),
                                 float(pose[2])],
                    "information_gain": 0.0, "score": 0.0,
                    "source": "mirrored_pair_station_recenter",
                    "scheduler_phase": "MIRRORED_PAIR_STATION_RECENTER",
                }
            self.local_candidate_attempts[key] = self.elapsed() - 40.0
            self.corridor_sweep_history.append({
                "event": "MIRRORED_PAIR_PORTAL_VALIDATION_REJECTED",
                "elapsed_sec": round(self.elapsed(), 3),
                "source_branch_id": branch.branch_id,
                "source_station_m": branch.station,
                "action": "bounded_corridor_rotation_then_fresh_door_detection",
            })
            # Stay on the verified corridor side.  A fast half-turn refreshes
            # the local raster and door-frame normal without commanding motion
            # through a wall that has not yet been observed as open.
            self._perform_local_rescan(
                angle_rad=math.pi, angular_speed=self.room_camera_sweep_speed,
                reason="mirrored_pair_portal_geometry_refresh",
                visual_sweep=True, room_id=None)
            return None
        goal = self.room_scheduler.next_goal(
            grid, (float(pose[0]), float(pose[1])), self.elapsed())
        if goal is None:
            return None
        goal["position"][2] = float(pose[2])
        goal["scheduler_phase"] = "MIRRORED_PAIRED_ROOM_VERIFY"
        goal["door_takeover_source"] = "verified_door_mirror_fallback"
        goal["local_door_candidate_id"] = key
        goal["local_door_evidence"] = evidence
        self.corridor_sweep_history.append({
            "event": "RETURN_PASS_MIRRORED_PAIRED_ROOM_SELECTED",
            "elapsed_sec": round(self.elapsed(), 3),
            "source_branch_id": branch.branch_id,
            "source_station_m": branch.station,
            "goal": list(evidence["entry_goal"]),
            "corridor_centerline_recenter_m": round(
                float(np.median(corridor_side_offsets)), 3)
                if corridor_side_offsets else 0.0,
            "safety": "visited-door symmetry plus occupancy A-star",
        })
        return goal

    def _plan_upper_floor_truth_corridor_probe(self, cycle):
        """Advance from a verified upper-floor ingress when the map is empty.

        The upper-floor projection can legitimately contain only the landing
        and the first corridor slice.  In that case ordinary frontier/A*
        planning has no target, even though the truth handoff has placed the
        robot safely on the corridor centreline.  Use a short, bounded
        truth-axis segment to expose the next slice; all room detection and
        subsequent navigation remain online.
        """
        if (not self.corridor_guided_entry_door_context or
                int(rospy.get_param("~floor_number", 1)) < 2 or
                not self.corridor_established or
                self.room_scheduler.active_door is not None):
            return None
        with self.lock:
            pose = self.pose
        if pose is None:
            return None
        axis = (np.asarray(self.corridor_station_axis, dtype=float)
                if self.corridor_station_axis is not None else
                np.asarray(self.corridor_axis, dtype=float)
                if self.corridor_axis is not None else
                np.asarray(self.corridor_forward_axis_hint, dtype=float)
                if self.corridor_forward_axis_hint is not None else None)
        if axis is None:
            return None
        axis /= max(1e-9, float(np.linalg.norm(axis)))
        origin = (np.asarray(self.corridor_station_origin, dtype=float)
                  if self.corridor_station_origin is not None else
                  np.asarray([float(pose[0]), float(pose[1])], dtype=float))
        current = np.asarray([float(pose[0]), float(pose[1])], dtype=float)
        centerline = origin + np.dot(current-origin, axis)*axis
        floor_num = int(rospy.get_param("~floor_number", 1))
        next_probe_index = int(getattr(
            self, "upper_floor_truth_probe_selected_count", 0)) + 1
        if (floor_num >= 3 and self.upper_floor_truth_probe_prescan and
                (next_probe_index - 1) %
                self.upper_floor_truth_probe_prescan_every == 0):
            # Run11 crossed the near F3 doorway row with a single moving
            # detector frame, then issued four 3.65 m truth segments before
            # accepting the far-row room. Pause at the current centreline
            # station, observe both walls, and give the ordinary online door
            # scheduler first refusal before advancing.
            self._perform_local_rescan(
                angle_rad=self.upper_floor_truth_probe_prescan_angle,
                angular_speed=self.room_camera_sweep_speed,
                reason="third_floor_door_first_corridor_observation")
            door_goal = self._plan_corridor_door_goal()
            if door_goal is None:
                door_goal = self._plan_local_room_entry_goal()
            if door_goal is None:
                door_goal = self._plan_known_unvisited_door_retry_goal()
            if door_goal is not None:
                self.corridor_sweep_history.append({
                    "event": "UPPER_FLOOR_TRUTH_PROBE_PREEMPTED_BY_DOOR",
                    "elapsed_sec": round(self.elapsed(), 3),
                    "cycle": int(cycle),
                    "probe_index": next_probe_index,
                    "door_id": door_goal.get("estimated_door_id"),
                    "source": door_goal.get("source"),
                })
                return door_goal
        # Upper-floor far room row: the row-2 doorways (station ~21-24 m)
        # must be positively observed from the corridor centre before the
        # outward pass proceeds.  A fast centreline transit can leave only a
        # few detector frames at each aperture (four_rooms_v6: 6 frames,
        # zero confirmed candidates -> row-2 rooms never entered).  Insert a
        # stationary 360-degree scan once per far-row crossing so the
        # detector accumulates enough observations to confirm both sides.
        station = float(np.dot(current - origin, axis))
        far_row_start = float(getattr(
            self, "corridor_far_row_scan_station_m",
            18.0 if floor_num >= 2 else 999.0))
        far_row_end = far_row_start + 6.0
        scan_key = "upper_far_row_scanned"
        if (floor_num >= 2 and
                far_row_start <= station <= far_row_end and
                not getattr(self, scan_key, False) and
                self.room_scheduler.active_door is None and
                self.enable_local_doorway_detector):
            # Only stop once per outward pass; the return pass is handled by
            # the normal door-preemptible supplement logic.
            setattr(self, scan_key, True)
            self._perform_local_rescan(
                angle_rad=self.upper_floor_far_row_door_scan_angle,
                angular_speed=self.room_camera_sweep_speed,
                reason="upper_floor_far_row_door_scan")
            self.corridor_sweep_history.append({
                "event": "UPPER_FLOOR_FAR_ROW_DOOR_SCAN",
                "elapsed_sec": round(self.elapsed(), 3),
                "station_m": round(station, 3),
                "policy": "stationary_scan_before_far_room_row",
            })
            # Fall through and still issue the forward probe so the outward
            # pass continues normally.
        # The probe must land strictly beyond the minimum-advance gate.  A
        # target exactly at corridor_forward_distance (2.5 m, equal to the
        # gate) can measure only 2.49 m of corridor progress and be rejected
        # as a backward goal (run ..._224: CORRIDOR_BACKWARD_GOAL_REJECTED
        # with original_progress_m=2.495), which latches a blocked-probe
        # return on the upper floor before any room is entered.  Target a
        # small margin past the gate, still inside the forward-goal
        # acceptance band (corridor_forward_distance + 0.25).
        configured_advance = float(getattr(
            self, "upper_floor_truth_probe_distance", 0.0))
        advance = (max(0.8, configured_advance)
                   if configured_advance > 0.0 else
                   max(0.8, float(self.corridor_forward_distance) + 0.15))
        target = centerline + advance*axis
        self.upper_floor_truth_probe_selected_count = next_probe_index
        goal = {
            "position": [float(target[0]), float(target[1]), float(pose[2])],
            "information_gain": 0.0,
            "score": 0.0,
            "source": "upper_floor_truth_corridor_probe",
            "scheduler_phase": "CORRIDOR_SWEEP",
            "corridor_truth_probe": True,
            "corridor_advance_m": advance,
            "execution_timeout_sec": 25.0,
            "_preplanned_path_result": {
                "success": True,
                "reason": "verified_upper_floor_truth_corridor_segment",
                "path": [(float(pose[0]), float(pose[1])),
                         (float(target[0]), float(target[1]))],
                "execution_waypoints": [(float(target[0]), float(target[1]))],
            },
        }
        self.corridor_sweep_history.append({
            "event": "UPPER_FLOOR_TRUTH_CORRIDOR_PROBE_SELECTED",
            "elapsed_sec": round(self.elapsed(), 3),
            "cycle": int(cycle),
            "start": [float(pose[0]), float(pose[1])],
            "target": [float(target[0]), float(target[1])],
            "axis": [float(axis[0]), float(axis[1])],
        })
        rospy.logwarn("Upper-floor truth corridor probe: (%.2f, %.2f) -> (%.2f, %.2f).",
                      pose[0], pose[1], target[0], target[1])
        return goal

    def _sweep_direction_centered_on_target(
            pose, target, angle_rad, fallback=1.0):
        """Choose the sign whose sweep midpoint faces a sensing target.

        The two physical room views are camera observations, not heading
        preparation for the next translation.  Aligning a sweep *endpoint*
        with G4/EXIT can make the primary and complementary fans overlap and
        leave the same inward sector unseen.  Centering both fans on the door
        inward normal covers the room-facing hemisphere without increasing
        either requested angle or its execution time.
        """
        try:
            if pose is None or target is None or len(target) < 2:
                raise ValueError("missing sweep alignment geometry")
            start_yaw = float(pose[3])
            target_x = float(target[0])
            target_y = float(target[1])
            half_angle = 0.5 * abs(float(angle_rad))
        except (IndexError, TypeError, ValueError):
            return -1.0 if float(fallback) < 0.0 else 1.0
        desired = math.atan2(target_y - float(pose[1]),
                             target_x - float(pose[0]))
        initial_error = math.atan2(
            math.sin(desired - start_yaw),
            math.cos(desired - start_yaw))
        # If the inward normal is already inside the initial RGB camera
        # cone, spend the body fan on the opposite sector instead.
        if abs(initial_error) <= math.radians(25.0):
            if abs(initial_error) <= math.radians(2.0):
                return -1.0 if float(fallback) < 0.0 else 1.0
            return -1.0 if initial_error > 0.0 else 1.0

        def midpoint_error(direction):
            midpoint = start_yaw + direction * half_angle
            return abs(math.atan2(math.sin(desired - midpoint),
                                  math.cos(desired - midpoint)))

        positive_error = midpoint_error(1.0)
        negative_error = midpoint_error(-1.0)
        fallback_direction = -1.0 if float(fallback) < 0.0 else 1.0
        if abs(positive_error - negative_error) <= 0.05:
            return fallback_direction
        return -1.0 if negative_error < positive_error else 1.0

    def _wait_for_visual_sweep_settle(self, room_id=None):

        """Bounded measured-motion gate before a fast rotation-only action."""
        maximum_wait = self.visual_sweep_settle_maximum_seconds
        if maximum_wait <= 0.0:
            return True
        started = time.monotonic()
        deadline = started + maximum_wait
        below_since = None
        last_linear = math.inf
        last_angular = math.inf
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            now = time.monotonic()
            with self.lock:
                last_linear = float(self.odom_linear_speed)
                last_angular = float(self.odom_angular_speed)
                sample_age = now - float(self.odom_speed_updated_at)
            settled = (sample_age <= 0.5 and
                       math.isfinite(last_linear) and
                       math.isfinite(last_angular) and
                       last_linear <= self.visual_sweep_settle_linear_speed and
                       last_angular <= self.visual_sweep_settle_angular_speed)
            if settled:
                if below_since is None:
                    below_since = now
                if now - below_since >= self.visual_sweep_settle_hold_seconds:
                    self.corridor_sweep_history.append({
                        "event": "ROOM_VISUAL_SWEEP_SETTLED",
                        "elapsed_sec": round(self.elapsed(), 3),
                        "room_id": room_id,
                        "waited_sec": round(now - started, 3),
                        "linear_speed_mps": round(last_linear, 3),
                        "angular_speed_rps": round(last_angular, 3),
                    })
                    return True
            else:
                below_since = None
            time.sleep(0.03)
        self.corridor_sweep_history.append({
            "event": "ROOM_VISUAL_SWEEP_SETTLE_TIMEOUT",
            "elapsed_sec": round(self.elapsed(), 3),
            "room_id": room_id,
            "waited_sec": round(time.monotonic() - started, 3),
            "linear_speed_mps": (round(last_linear, 3)
                                 if math.isfinite(last_linear) else None),
            "angular_speed_rps": (round(last_angular, 3)
                                  if math.isfinite(last_angular) else None),
        })
        return False

    def request_external_abort(self, reason):
        """Stop an active mission when its supervising floor guard trips."""
        reason = str(reason or "EXTERNAL_ABORT")
        with self.lock:
            if self.external_abort_reason is None:
                self.external_abort_reason = reason
        self.cancel_goal_pub.publish(String(
            data="external_mission_abort:" + reason))

    def _spawn_shutdown_visualization(self):
        """Detach post-processing so Ctrl-C cannot kill it with this node."""
        if self.shutdown_visualization_spawned:
            return
        self.shutdown_visualization_spawned = True
        floor_slug = os.path.basename(os.path.normpath(self.output_dir))
        run_dir = (os.path.dirname(os.path.normpath(self.output_dir))
                   if floor_slug in ("second_floor", "third_floor") else
                   self.output_dir)
        if (not os.path.isdir(run_dir) or
                not os.path.isfile(self.visualization_script)):
            rospy.logerr("Shutdown visualization inputs missing: run=%s script=%s",
                         run_dir, self.visualization_script)
            return
        command = [
            sys.executable, self.visualization_script,
            "--run-dir", run_dir,
            "--output-dir", os.path.join(run_dir, "visualization"),
        ]
        try:
            subprocess.Popen(
                command, stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                close_fds=True, start_new_session=True)
            rospy.loginfo(
                "Detached shutdown visualization scheduled for %s; "
                "figures will continue after Ctrl-C.", run_dir)
        except OSError as error:
            rospy.logerr("Could not start shutdown visualization: %s", error)

    def _start_localization_truth_recovery(self):
        now = self.elapsed()
        with self.lock:
            self._localization_recovery_window.reset()
            self._localization_recovery_window.begin(
                now, self._fastlio_registration_status_sequence,
                self.grid_update_count)
            self._localization_recovery_exhausted = False
            self._sync_localization_recovery_diagnostics_locked()
        return now

    def _advance_localization_truth_recovery(self, invalid_count,
                                             request_reason):
        now = self.elapsed()
        with self.lock:
            window = self._localization_recovery_window
            outcome = window.poll(now, self.grid_update_count)
            attempt = int(window.attempts)
            started_at = float(window.started_at)
            healthy_streak = int(window.healthy_streak)
            fresh_grid_updates = max(
                0, int(self.grid_update_count) -
                int(window.grid_update_at_start))
            self._sync_localization_recovery_diagnostics_locked()

        if outcome == LocalizationRecoveryWindow.REQUEST:
            reason = (str(request_reason) if attempt == 1 else
                      "registration_unhealthy_retry")
            self._publish_truth_recovery_request(reason, invalid_count)
            self.corridor_sweep_history.append({
                "event": ("LOCALIZATION_TRUTH_RECOVERY_REQUESTED"
                          if attempt == 1 else
                          "LOCALIZATION_TRUTH_RECOVERY_RETRY"),
                "elapsed_sec": round(now, 3),
                "invalid_count": int(invalid_count),
                "attempt": attempt,
                "grace_sec": self.localization_truth_recovery_grace_sec,
                "retry_interval_sec":
                    self.localization_truth_recovery_retry_interval_sec,
                "policy": "single_fixed_deadline_wait_for_reanchor",
            })
            return outcome

        if outcome == LocalizationRecoveryWindow.RECOVERED:
            self.corridor_sweep_history.append({
                "event": "LOCALIZATION_TRUTH_RECOVERY_CONFIRMED",
                "elapsed_sec": round(now, 3),
                "recovery_elapsed_sec": round(now - started_at, 3),
                "attempts": attempt,
                "healthy_confirmations": healthy_streak,
                "fresh_grid_updates": fresh_grid_updates,
                "policy": "post_request_health_and_map_confirmed",
            })
            with self.lock:
                self._localization_recovery_window.reset()
                self._localization_recovery_exhausted = False
                self._sync_localization_recovery_diagnostics_locked()
            return outcome

        if outcome == LocalizationRecoveryWindow.EXHAUSTED:
            self.corridor_sweep_history.append({
                "event": "LOCALIZATION_TRUTH_RECOVERY_EXHAUSTED",
                "elapsed_sec": round(now, 3),
                "recovery_elapsed_sec": round(now - started_at, 3),
                "invalid_count": int(invalid_count),
                "attempts": attempt,
                "healthy_confirmations": healthy_streak,
                "fresh_grid_updates": fresh_grid_updates,
                "grace_sec": self.localization_truth_recovery_grace_sec,
            })
            with self.lock:
                self._localization_recovery_window.reset()
                self._localization_recovery_exhausted = True
                self._sync_localization_recovery_diagnostics_locked()
            return outcome
        return outcome

    def _localization_truth_recovery_in_progress(self, registration_healthy,
                                                  registration_invalid_count):
        """Hold planning until one bounded recovery episode resolves."""
        if not self.localization_truth_recovery_enabled:
            return False
        with self.lock:
            active = self._localization_recovery_window.active
            exhausted = self._localization_recovery_exhausted
        if not active:
            if (exhausted or registration_healthy or
                    registration_invalid_count <
                    self.localization_truth_recovery_start_invalid_count):
                return False
            self._start_localization_truth_recovery()
        outcome = self._advance_localization_truth_recovery(
            registration_invalid_count, "registration_unhealthy")
        return outcome in (
            LocalizationRecoveryWindow.REQUEST,
            LocalizationRecoveryWindow.WAITING)

    def _wait_for_active_goal_localization_recovery(
            self, execution_result, cycle, goal):
        """Stop, recover and require fresh localization/map before replanning."""
        if not self.localization_truth_recovery_enabled:
            return False
        cancel_payload = structured_executor_cancel(
            execution_result, "localization_truth_recovery_hold")
        if cancel_payload is not None:
            self.cancel_goal_pub.publish(String(
                data=json.dumps(cancel_payload, sort_keys=True)))
        started_at = self._start_localization_truth_recovery()
        self.corridor_sweep_history.append({
            "event": "ACTIVE_GOAL_LOCALIZATION_RECOVERY_HOLD",
            "elapsed_sec": round(started_at, 3),
            "goal_id": int(cycle),
            "goal_source": goal.get("source"),
            "room_role": goal.get("room_role"),
            "structured_cancel_published": cancel_payload is not None,
            "stop_confirmation":
                "executor_result_is_published_after_locked_zero_command",
            "deadline_sec": self.localization_truth_recovery_grace_sec,
        })

        def finish_room_budget_pause():
            duration = max(0.0, self.elapsed() - started_at)
            room_started_at = self.room_scheduler.room_started_at
            if room_started_at is not None:
                self.room_scheduler.room_started_at = (
                    float(room_started_at) + duration)
                self.corridor_sweep_history.append({
                    "event":
                        "ROOM_BUDGET_EXCLUDED_LOCALIZATION_RECOVERY",
                    "elapsed_sec": round(self.elapsed(), 3),
                    "pause_duration_sec": round(duration, 3),
                    "room_id": self.room_scheduler.active_room_id,
                    "policy": "mission_clock_unchanged_room_clock_frozen",
                })
            return duration

        while not rospy.is_shutdown():
            with self.lock:
                invalid_count = self.fastlio_registration_invalid_count
            outcome = self._advance_localization_truth_recovery(
                invalid_count, "registration_unhealthy_active_goal")
            if outcome == LocalizationRecoveryWindow.RECOVERED:
                finish_room_budget_pause()
                return True
            if outcome == LocalizationRecoveryWindow.EXHAUSTED:
                finish_room_budget_pause()
                return False
            time.sleep(0.05)
        with self.lock:
            self._localization_recovery_window.reset()
            self._sync_localization_recovery_diagnostics_locked()
        finish_room_budget_pause()
        return False

    def _post_exit_opposite_centerline_exception(
            self, candidate_center, candidate_lateral):
        """Admit only the pending same-station opposite aperture in the band.

        This is a narrow online-geometry exception for localization drift.  It
        never treats the corridor centre itself as a door: the candidate must
        lie on the opposite signed side, at the just-exited station, and retain
        a measurable lateral displacement.  The ordinary confirmation, A*,
        SCAN-lite and portal gates still run afterwards.
        """
        if (self.last_room_exit_side is None or
                self.last_room_exit_pose is None or
                candidate_center is None):
            return False
        with self.lock:
            pose = self.pose
        if not self._recent_exit_needs_opposite_check(pose):
            return False
        axis = np.asarray(self.corridor_station_axis, dtype=float)
        norm = float(np.linalg.norm(axis))
        if norm < 1e-6:
            return False
        axis /= norm
        normal = np.asarray([-axis[1], axis[0]], dtype=float)
        origin = np.asarray(self.corridor_station_origin, dtype=float)
        center = np.asarray(candidate_center[:2], dtype=float)
        signed_lateral = float(np.dot(center - origin, normal))
        candidate_side = 1 if signed_lateral >= 0.0 else -1
        if candidate_side != -int(self.last_room_exit_side):
            return False
        minimum = float(getattr(
            self, "room_door_minimum_centerline_lateral", 0.0))
        if float(candidate_lateral) < max(0.08, 0.15 * minimum):
            return False
        candidate_station = self._corridor_station(center)
        exit_station = self._corridor_station(self.last_room_exit_pose)
        return bool(abs(float(candidate_station) - float(exit_station)) <=
                    self.corridor_branch_station_tolerance)

    def _multi_floor_truth_handoff_safe(self):
        """Relaxed safety gate for the TIME_LIMIT/GOAL_FAILURE truth fallback.

        The strict _stair_return_handoff_authorized() requires a physical
        terminal-return latch, which a TIME_LIMIT at the corridor mid-point
        never satisfies (run ..._231: exited=2, no latch).  The truth stair
        route begins from the robot's current pose and can safely traverse the
        corridor, so the fallback only needs to prove the dog is on this
        floor's flat corridor (not inside a room or mid-stair) and that the
        floor mission actually started.  This prevents a silent F1 exit that
        leaves the stair manager hanging in STAIR_HANDOFF_NOT_REACHED.
        """
        if not self._floor_handoff_room_requirement_met():
            exited_rooms = sum(
                bool(door.visited)
                for door in self.room_scheduler.detector.doors)
            # 松弛闸内部不再套严格闸: strict 4/4 是探索中的质量门, TIME_LIMIT/
            # GOAL_FAILURE 兜底时房间不足也应继续 —— 否则 3/4 时松弛闸被严格闸
            # 挡死, 交接永不触发(批次16 RUN2/3 STAIR_HANDOFF_NOT_REACHED 根因,
            # A 版内部矛盾: 松弛闸第一道检查调用严格闸)。后续 pose/active_door/
            # z 检查已保证不在房间内、处于平坦走廊。
            rospy.logwarn(
                "multi_floor_truth_handoff_safe: relaxed physical exits "
                "%d/%d (strict 4/4 gate skipped for truth fallback)",
                exited_rooms, max(4, int(self.room_target_count)))
        with self.lock:
            pose = self.pose
        if pose is None:
            rospy.logwarn("multi_floor_truth_handoff_safe: FAIL pose=None")
            return False
        if self.room_scheduler.active_door is not None:
            rospy.logwarn("multi_floor_truth_handoff_safe: FAIL active_door=%s",
                          self.room_scheduler.active_door.door_id)
            return False
        entered_any_room = bool(
            self.room_scheduler.detector.doors and
            any(bool(door.visited)
                for door in self.room_scheduler.detector.doors))
        # A floor that never entered any room should not jump to the stairs
        # without at least one observed doorway.
        if not entered_any_room and not self.corridor_established:
            rospy.logwarn(
                "multi_floor_truth_handoff_safe: FAIL no rooms and "
                "corridor_established=%s", self.corridor_established)
            return False
        # Require the body to be on a plausible flat floor (not mid-stair or
        # inside a fallen pose).  z is expressed in the mapper's planar frame
        # which is near the base height on the current floor.
        z = float(pose[2])
        # The mapper planar frame sits at the base height, so a healthy
        # flat-floor odom z hovers around 0.0, not 0.05+.  run ..._850s FAILed
        # with z=-0.001 on a perfectly flat corridor.  Use a tolerant lower
        # bound and keep the 1.5 m ceiling to reject mid-stair/fallen poses.
        if not (-0.25 <= z <= 1.5):
            rospy.logwarn("multi_floor_truth_handoff_safe: FAIL z=%.3f", z)
            return False
        rospy.logwarn(
            "multi_floor_truth_handoff_safe: PASS entered_any_room=%s "
            "corridor_established=%s pose=(%.2f,%.2f,%.2f)",
            entered_any_room, self.corridor_established,
            pose[0], pose[1], z)
        return True

    def run(self):
        if not self._wait_slam():
            self.termination_reason = "WAIT_FOR_SLAM_TIMEOUT"
        elif not self._wait_map():
            self.termination_reason = "WAIT_FOR_MAP_TIMEOUT"
        else:
            # Startup establishes ROS, FAST-LIO and the first occupancy map;
            # it is not exploration work.  Start the user-facing mission
            # budget only once those prerequisites are ready, while retaining
            # the elapsed startup duration for offline diagnostics.
            self.startup_elapsed_sec = self.elapsed()
            self.started = time.monotonic()
            self.mission_clock_started = True
            with self.lock:
                # Startup poses are stationary prerequisites, not part of the
                # exploration trajectory or its room-exit snapshots.
                self.trajectory.clear()
            rospy.loginfo("[STARTUP] Complete in %.2fs (excluded from mission budget)",
                          self.startup_elapsed_sec)
            rospy.loginfo(
                "[MISSION] CLOCK START t=0.000/%.1fs; startup excluded; "
                "subsequent [MISSION] t=... is the only task timer",
                self.maximum_duration)
            self._set_state("MISSION_CLOCK_START", "startup_complete")
        cycle = 0
        consecutive_failures = 0
        while not rospy.is_shutdown() and self.termination_reason is None:
            # Scheduler transitions can happen inside a completed goal
            # callback, between named manager states. Refresh the latched
            # RGB-D room gate on every planning cycle as well.
            self._publish_room_detection_context()
            # [PORTED zip140s A:10688-10733] 定位真值恢复闸: 注册丢失时先走
            # 有界恢复窗口, 恢复窗口进行中则睡眠续转; 恢复耗尽/不可恢复才 abort。
            # 整体以 hasattr 容错: 恢复子系统未移植前此闸门为 no-op, B 现状保留。
            if hasattr(self, "_localization_truth_recovery_in_progress"):
                with self.lock:
                    registration_healthy = self.fastlio_registration_healthy
                    registration_invalid_count = self.fastlio_registration_invalid_count
                    localization_recovery_active = getattr(
                        self, "_localization_recovery_active", False)
                if (localization_recovery_active or
                        (not registration_healthy and
                         registration_invalid_count >= getattr(
                             self,
                             "localization_truth_recovery_start_invalid_count",
                             3))):
                    if self._localization_truth_recovery_in_progress(
                            registration_healthy, registration_invalid_count):
                        rospy.sleep(0.05)
                        continue
                # The recovery poll may have consumed callbacks newer than the
                # snapshot above.  Re-read all registration fields so a
                # confirmed recovery cannot be followed by an abort based on
                # its stale pre-recovery invalid count.
                with self.lock:
                    registration_healthy = self.fastlio_registration_healthy
                    registration_invalid_count = self.fastlio_registration_invalid_count
                    localization_recovery_exhausted = getattr(
                        self, "_localization_recovery_exhausted", False)
                if (not getattr(
                        self,
                        "allow_truth_exploration_when_registration_lost",
                        False) and
                        (localization_recovery_exhausted or
                         (not registration_healthy and
                          registration_invalid_count >=
                          self.fastlio_registration_abort_invalid_count))):
                    self.termination_reason = "FASTLIO_REGISTRATION_LOST"
                    self.corridor_sweep_history.append({
                        "event": "MISSION_ABORT_FASTLIO_REGISTRATION_LOST",
                        "elapsed_sec": round(self.elapsed(), 3),
                        "invalid_count": registration_invalid_count,
                        "recovery_exhausted":
                            bool(localization_recovery_exhausted),
                        "action": "stop_motion_finalize_logs_and_visualization",
                    })
                    self.cancel_goal_pub.publish(String(
                        data="fastlio_registration_lost_mission_abort"))
                    break
                if (not registration_healthy and
                        not getattr(
                            self,
                            "allow_truth_exploration_when_registration_lost",
                            False) and
                        not getattr(
                            self, "_localization_recovery_active", False)):
                    # F1 holds scheduling while FAST-LIO retries from its
                    # trusted anchor.  Upper floors may continue only after a
                    # truth-guided handoff has explicitly enabled this
                    # bounded fallback.
                    rospy.sleep(0.05)
                    continue
            with self.lock:
                registration_healthy = self.fastlio_registration_healthy
                registration_invalid_count = self.fastlio_registration_invalid_count
            if (not registration_healthy and
                    registration_invalid_count >=
                    self.fastlio_registration_abort_invalid_count):
                self.termination_reason = "FASTLIO_REGISTRATION_LOST"
                self.corridor_sweep_history.append({
                    "event": "MISSION_ABORT_FASTLIO_REGISTRATION_LOST",
                    "elapsed_sec": round(self.elapsed(), 3),
                    "invalid_count": registration_invalid_count,
                    "action": "stop_motion_finalize_logs_and_visualization",
                })
                self.cancel_goal_pub.publish(String(
                    data="fastlio_registration_lost_mission_abort"))
                break
            if not registration_healthy:
                # Hold scheduling briefly while FAST-LIO retries from its
                # trusted anchor.  A single delayed high-speed scan must not
                # terminate the mission, but no new goal is issued from a
                # temporarily frozen pose either.
                rospy.sleep(0.05)
                continue
            if self.elapsed() >= self._stair_return_deadline():
                self.termination_reason = "TIME_LIMIT"
                break
            if self.maximum_goals > 0 and len(self.goal_history) >= self.maximum_goals:
                self.termination_reason = "TIME_LIMIT"
                break
            with self.lock:
                scheduling_pose = self.pose
            self.depth_breadth.ensure_started(
                (scheduling_pose[0], scheduling_pose[1]), self.elapsed())
            with self.lock:
                scheduling_grid = self.grid
            region_context = self._corridor_context(
                scheduling_pose, scheduling_grid)
            region_is_corridor = self._on_established_corridor_centerline(
                scheduling_pose, region_context)
            if region_is_corridor:
                self.last_region_corridor_pose = (
                    float(scheduling_pose[0]), float(scheduling_pose[1]))
                if self.previous_region_was_corridor is False:
                    room_coverage_complete = bool(
                        self.room_scheduler.detector.doors and
                        self.room_scheduler.detector.doors[-1].visited and
                        self.room_scheduler.detector.doors[-1].coverage_complete)
                    branch_state = self.branch_scheduler.mark_exit(
                        self.elapsed(),
                        room_coverage_complete or
                        self.depth_breadth.escape_reason ==
                        "coverage_complete")
                    if branch_state is not None:
                        self.depth_breadth_history.append({
                            "event": "CORRIDOR_BRANCH_EXIT",
                            "elapsed_sec": round(self.elapsed(), 3),
                            "branch_state": branch_state,
                            "coverage_complete":
                                room_coverage_complete or
                                self.depth_breadth.escape_reason ==
                                "coverage_complete",
                            "coverage_source": (
                                "lightweight_room_lidar" if
                                room_coverage_complete else
                                "region_coverage"),
                        })
                    self.depth_breadth.mark_corridor_transit()
                    self.active_region_side_sign = None
                    self.post_entry_observed_cells.clear()
                    self.region_exit_recovery_attempts = 0
                    self.region_return_failed_targets.clear()
                    self.region_stall_backtrack_pending = False
                    self.region_stall_backtrack_attempts = 0
            elif self.previous_region_was_corridor is True:
                entry_pose = (self.last_region_corridor_pose or
                              (float(scheduling_pose[0]),
                               float(scheduling_pose[1])))
                axis = np.asarray(
                    self.corridor_station_axis
                    if self.corridor_station_axis is not None else
                    region_context["axis"], dtype=float)
                normal = np.asarray([-axis[1], axis[0]], dtype=float)
                center = np.asarray(
                    region_context["centerline_point"], dtype=float)
                lateral = float(np.dot(
                    np.asarray(scheduling_pose[:2], dtype=float) - center,
                    normal))
                self.active_region_side_sign = (
                    1.0 if lateral >= 0.0 else -1.0)
                component = self._visited_side_component(
                    scheduling_pose[:2],
                    int(self.active_region_side_sign), scheduling_grid)
                event = self.depth_breadth.mark_region_entry(
                    entry_pose,
                    (float(scheduling_pose[0]), float(scheduling_pose[1])),
                    self.elapsed())
                branch = self.branch_scheduler.register(
                    self._corridor_station(entry_pose),
                    int(self.active_region_side_sign),
                    (float(scheduling_pose[0]), float(scheduling_pose[1])),
                    self.elapsed())
                matched_ids = component.get("matched_branch_ids", [])
                if component.get("matched") and matched_ids:
                    branch = self.branch_scheduler.merge_ids(
                        matched_ids[0], branch.branch_id) or branch
                self.branch_scheduler.mark_entered(branch.branch_id)
                if component.get("matched"):
                    event = self.depth_breadth.request_entry_return(
                        "visited_region_reentry")
                self._remember_visited_region_sample(
                    scheduling_pose[:2], int(self.active_region_side_sign),
                    branch.branch_id)
                self.post_entry_observed_cells.clear()
                self.region_exit_recovery_attempts = 0
                self.region_return_failed_targets.clear()
                self.region_stall_backtrack_pending = False
                self.region_stall_backtrack_attempts = 0
                self.depth_breadth_history.append({
                    "event": ("VISITED_REGION_REENTRY" if
                              component.get("matched") else
                              "PHYSICAL_REGION_ENTRY"),
                    "result": event,
                    "elapsed_sec": round(self.elapsed(), 3),
                    "region_index": self.depth_breadth.region_index,
                    "entry_pose": entry_pose,
                    "inside_pose": [float(scheduling_pose[0]),
                                    float(scheduling_pose[1])],
                    "corridor_branch_id": branch.branch_id,
                    "corridor_branch_station_m": branch.station,
                    "corridor_branch_side": branch.side,
                    "component_cells": component.get("component_cells"),
                    "matched_branch_ids": matched_ids,
                })
            elif (self.previous_region_was_corridor is False and
                  self.active_region_side_sign is not None and
                  region_context is not None):
                # A single high-level path can cross the corridor and finish
                # in the opposite side area without a scheduling sample on
                # the centreline. Detect the sign change geometrically and
                # start fresh entry/coverage evidence for the new side.
                axis = np.asarray(
                    self.corridor_station_axis
                    if self.corridor_station_axis is not None else
                    region_context["axis"], dtype=float)
                normal = np.asarray([-axis[1], axis[0]], dtype=float)
                center = np.asarray(
                    region_context["centerline_point"], dtype=float)
                current = np.asarray(scheduling_pose[:2], dtype=float)
                signed_lateral = float(np.dot(current - center, normal))
                current_sign = 1.0 if signed_lateral >= 0.0 else -1.0
                if (abs(signed_lateral) >
                        self.corridor_centerline_membership_tolerance and
                        current_sign != self.active_region_side_sign):
                    previous_state = self.branch_scheduler.mark_exit(
                        self.elapsed(),
                        self.depth_breadth.escape_reason ==
                        "coverage_complete")
                    entry = center
                    component = self._visited_side_component(
                        current, int(current_sign), scheduling_grid)
                    self.depth_breadth.mark_region_entry(
                        (float(entry[0]), float(entry[1])),
                        (float(current[0]), float(current[1])),
                        self.elapsed())
                    self.active_region_side_sign = current_sign
                    branch = self.branch_scheduler.register(
                        self._corridor_station(entry),
                        int(current_sign),
                        (float(current[0]), float(current[1])),
                        self.elapsed())
                    matched_ids = component.get("matched_branch_ids", [])
                    if component.get("matched") and matched_ids:
                        branch = self.branch_scheduler.merge_ids(
                            matched_ids[0], branch.branch_id) or branch
                    self.branch_scheduler.mark_entered(branch.branch_id)
                    if component.get("matched"):
                        self.depth_breadth.request_entry_return(
                            "visited_region_reentry")
                    self._remember_visited_region_sample(
                        current, int(current_sign), branch.branch_id)
                    self.post_entry_observed_cells.clear()
                    self.region_exit_recovery_attempts = 0
                    self.region_return_failed_targets.clear()
                    self.region_stall_backtrack_pending = False
                    self.region_stall_backtrack_attempts = 0
                    self.depth_breadth_history.append({
                        "event": ("VISITED_REGION_REENTRY" if
                                  component.get("matched") else
                                  "OPPOSITE_SIDE_REGION_ENTRY"),
                        "elapsed_sec": round(self.elapsed(), 3),
                        "region_index": self.depth_breadth.region_index,
                        "entry_pose": [float(entry[0]), float(entry[1])],
                        "inside_pose": [float(current[0]), float(current[1])],
                        "previous_branch_state": previous_state,
                        "corridor_branch_id": branch.branch_id,
                        "corridor_branch_station_m": branch.station,
                        "corridor_branch_side": branch.side,
                        "component_cells": component.get("component_cells"),
                        "matched_branch_ids": matched_ids,
                    })
            elif self.previous_region_was_corridor is None:
                # A mission may start in a wide area.  Treat the initial pose
                # as its entry, but still require subsequent inward motion and
                # at least one completed post-entry sensing goal.
                self.depth_breadth.mark_region_entry(
                    (float(scheduling_pose[0]), float(scheduling_pose[1])),
                    (float(scheduling_pose[0]), float(scheduling_pose[1])),
                    self.elapsed())
            self.previous_region_was_corridor = region_is_corridor
            if (not region_is_corridor and
                    self.active_region_side_sign is not None):
                self._remember_visited_region_sample(
                    scheduling_pose[:2],
                    int(self.active_region_side_sign),
                    self.branch_scheduler.active_branch_id)
            if (scheduling_grid is not None and
                    self.depth_breadth.anchor is not None and
                    not region_is_corridor):
                self._accumulate_post_entry_lidar(
                    scheduling_grid, scheduling_pose)
                coverage = local_region_coverage(
                    scheduling_grid, self.depth_breadth.anchor,
                    self.depth_breadth.config.region_radius,
                    self.post_entry_observed_cells)
                coverage_triggered = self.depth_breadth.update_coverage(
                    coverage, (scheduling_pose[0], scheduling_pose[1]),
                    self.elapsed())
                self.depth_breadth_history.append({
                    "event": ("REGION_COVERAGE_COMPLETE" if coverage_triggered
                              else "REGION_COVERAGE_UPDATED"),
                    "elapsed_sec": round(self.elapsed(), 3),
                    "region_index": self.depth_breadth.region_index,
                    "region_anchor": self.depth_breadth.anchor,
                    "region_entry_anchor": self.depth_breadth.entry_anchor,
                    "coverage": self.depth_breadth.coverage_status,
                    "coverage_streak": self.depth_breadth.coverage_streak,
                    "entry_return_pending":
                        self.depth_breadth.entry_return_pending,
                })
            if self.depth_breadth.exclusion(self.elapsed()) is not None:
                self._set_state("DEPTH_BREADTH_GLOBAL_SWITCH",
                                self.depth_breadth.escape_reason)
            cycle += 1
            rooms_complete = self._mission_rooms_complete()
            # Returning home is gated by the online corridor terminal wall.
            # Room-count completion alone is deliberately insufficient.
            with self.lock:
                gate_pose, gate_grid = self.pose, self.grid
            gate_context = (self._corridor_context(gate_pose, gate_grid)
                            if gate_pose is not None and gate_grid is not None
                            else None)
            gate_axis = (self.corridor_axis if self.corridor_axis is not None
                         else (gate_context.get("axis") if gate_context else None))
            exited_rooms = sum(bool(door.visited)
                               for door in self.room_scheduler.detector.doors)
            all_rooms_exited = (self.room_scheduler.active_door is None and
                                exited_rooms >= self.room_target_count)
            # A confirmed terminal wall starts the return pass even if some
            # rooms were missed on the outward pass.  Door candidates remain
            # eligible during that pass, so it is a bounded opportunity to
            # recover missed rooms rather than spending the whole budget
            # rescanning the wall at the corridor end.
            # A nearby cross-wall/room corner can look terminal before the
            # far doorway pair has entered the local map.  Require a sizeable
            # forward separation from every known doorway station before a
            # wall observation is allowed to start the return pass.
            known_stations=[float(branch.station) for branch in
                            self.branch_scheduler.branches]
            station_span=(max(known_stations) - min(known_stations)
                          if len(known_stations) >= 2 else 0.0)
            current_station=(float(self.corridor_forward_station_sign or 1.0) *
                             self._corridor_station(gate_pose)
                             if gate_pose is not None and self.corridor_station_axis is not None
                             else -math.inf)
            # An empty branch list means that corridor door discovery has not
            # happened yet; it does *not* mean every doorway is behind us.
            # Treating it as a vacuous "past all stations" condition caused a
            # newly entered corridor to return to the lobby after its first
            # short acquisition move.  A terminal-wall return is valid only
            # after at least one geometry-confirmed doorway station has been
            # recorded, and only well beyond the farthest such station.
            # A wall beside the first doorway pair can look like a corridor
            # terminus.  Before accepting a terminal wall on a partial pass,
            # require doorway evidence from two *separated* corridor stations
            # (or completion of all required rooms).  This is derived solely
            # from online door geometry, not room count/layout metadata, and
            # prevents a first-pair cross-wall from starting the return pass.
            terminal_structure_ok=(all_rooms_exited or station_span >= 8.0)
            # Four metres is enough to clear the far-room doorway in this
            # corridor while still leaving the final wall observable.
            terminal_progress_ok=(bool(known_stations) and
                                  terminal_structure_ok and
                                  current_station >= max(known_stations) + 4.0)
            terminal_wall_observed = bool(
                terminal_progress_ok and
                self.corridor_established and
                # At the real end wall, the forward opening is deliberately
                # absent and the local aspect-ratio classifier often stops
                # labelling the window as an open corridor.  Corridor
                # establishment plus the recorded doorway stations already
                # supplies the context; requiring is_corridor here rejected
                # repeated positive LiDAR wall observations at the terminus.
                self._forward_corridor_wall_observed(
                    gate_pose, gate_grid, gate_axis,
                    maximum_distance=self.corridor_terminal_return_distance))
            blocking_known_doors = self._known_unvisited_room_doors()
            terminal_known_door_block = bool(
                terminal_wall_observed and
                self._terminal_return_has_blocking_known_door(
                    all_rooms_exited))
            corridor_end_gate = bool(
                terminal_wall_observed and not terminal_known_door_block)
            # [PORTED zip140s A:11066-11072] 确认门否决: 机器人与标称终点墙带
            # 之间的真实门对证明走廊未结束, 墙后是下一排房间, 不是走廊终点。
            # 原始 f2_f3_fix_2 在 y=26.5 处提前返回, 从未到达 y=28.6 的 row-2 门。
            forward_door_ahead = False
            _confirmed_door_ahead_port = getattr(
                self, "_confirmed_local_door_ahead", None)
            if (_confirmed_door_ahead_port is not None and
                    gate_pose is not None and gate_axis is not None):
                forward_door_ahead = bool(_confirmed_door_ahead_port(
                    gate_pose, gate_grid, gate_axis,
                    lookahead=max(self.corridor_terminal_return_distance,
                                  10.0)))
            if forward_door_ahead and corridor_end_gate:
                corridor_end_gate = False
                self.corridor_sweep_history.append({
                    "event": "TERMINAL_RETURN_VETOED_CONFIRMED_DOOR_AHEAD",
                    "elapsed_sec": round(self.elapsed(), 3),
                    "policy": "confirmed_door_between_robot_and_terminal_wall",
                })
            if (terminal_known_door_block and
                    not self.terminal_known_door_block_logged):
                self.terminal_known_door_block_logged = True
                self.corridor_sweep_history.append({
                    "event": "TERMINAL_RETURN_WITHHELD_KNOWN_UNVISITED_DOOR",
                    "elapsed_sec": round(self.elapsed(), 3),
                    "door_ids": [door.door_id
                                 for door in blocking_known_doors],
                    "exited_rooms": exited_rooms,
                    "policy": "retry_known_door_before_stair_return",
                })
            if corridor_end_gate and not self.corridor_terminal_return_latched:
                self.corridor_terminal_return_latched = True
                self.corridor_sweep_history.append({
                    "event": "CORRIDOR_TERMINAL_RETURN_LATCHED",
                    "elapsed_sec": round(self.elapsed(), 3),
                    "wall_distance_limit_m": self.corridor_terminal_return_distance,
                    "known_station_span_m": round(station_span, 3),
                    "current_station_m": round(current_station, 3),
                    "exited_rooms": exited_rooms,
                })
            partial_return_reason = (
                self._partial_corridor_return_trigger()
                if (not all_rooms_exited and
                    self.room_scheduler.active_door is None and
                    self.stair_handoff_on_corridor_exit) else None)
            if (partial_return_reason and
                    not self.corridor_terminal_return_latched):
                self.corridor_terminal_return_latched = True
                self.corridor_partial_return_trigger_reason = \
                    partial_return_reason
                self.room_scheduler.config.door_cooldown_seconds = max(
                    self.room_scheduler.config.door_cooldown_seconds,
                    self.corridor_partial_return_reserve)
                self.corridor_sweep_history.append({
                    "event": "PARTIAL_FLOOR_RETURN_LATCHED",
                    "elapsed_sec": round(self.elapsed(), 3),
                    "trigger": partial_return_reason,
                    "forward_high_water_m": self.corridor_forward_station_high_water,
                    "remaining_mission_sec": round(
                        max(0.0, self.maximum_duration - self.elapsed()), 3),
                    "exited_rooms": exited_rooms,
                    "failed_door_cooldown_sec":
                        self.room_scheduler.config.door_cooldown_seconds,
                    "policy": "return_with_local_door_supplement_then_next_floor",
                })
            if all_rooms_exited and gate_pose is not None and gate_grid is not None \
                    and gate_axis is not None and gate_context \
                    and gate_context.get("is_corridor"):
                _, forward_probe_path = self._corridor_forward_plan(
                    cycle, gate_pose, gate_grid, gate_axis)
                if forward_probe_path is None:
                    self.post_exit_forward_blocked_streak += 1
                else:
                    self.post_exit_forward_blocked_streak = 0
                self.corridor_sweep_history.append({
                    "event": "POST_ROOM_FORWARD_BLOCKED_PROBE",
                    "elapsed_sec": round(self.elapsed(), 3),
                    "streak": self.post_exit_forward_blocked_streak,
                    "wall_observed": corridor_end_gate,
                })
            else:
                self.post_exit_forward_blocked_streak = 0
            if (self.stop_after_all_room_exits and all_rooms_exited and
                    (corridor_end_gate or
                     self.post_exit_forward_blocked_streak >= 2)):
                self.termination_reason = "ALL_ROOMS_EXITED_CORRIDOR_END"
                self._set_state("ROOM_EXITS_COMPLETE_CORRIDOR_END")
                self.corridor_sweep_history.append({
                    "event": "MISSION_STOP_AFTER_ROOM_EXITS",
                    "elapsed_sec": round(self.elapsed(), 3),
                    "exited_rooms": exited_rooms,
                    "terminal_wall_distance_m":
                        self.room_exit_terminal_wall_distance,
                    "reason": ("all_rooms_exited_and_forward_wall_observed"
                               if corridor_end_gate else
                               "all_rooms_exited_and_forward_path_blocked"),
                })
                break
            # A semantic-completion flag alone is insufficient: an entry may
            # be recognised while its physical exit is still pending.  A
            # rearward return is legal only after every required room has a
            # confirmed exit and the forward terminal wall is observed.
            # Once every required room has a confirmed physical EXIT there is
            # no remaining first-floor reason to spend time driving toward
            # the terminal wall.  For the stair-handoff mission, return from
            # the fourth exit directly to the latched corridor entrance.  If
            # a room is still missing, retain the stronger wall-gated return
            # so the reverse pass can still recover it.
            complete_room_direct_return = bool(
                self.stair_handoff_on_corridor_exit and all_rooms_exited)
            terminal_return_active = bool(
                self.corridor_terminal_return_latched and
                self.room_scheduler.active_door is None)
            returning_home = bool(
                self.room_scheduler.active_door is None and
                (complete_room_direct_return or terminal_return_active))
            if complete_room_direct_return or terminal_return_active:
                # Arm on either four exits or a confirmed terminal return.
                # During an incomplete return F1 still owns motion and its
                # long path remains preemptible by missed side doors. The
                # truth gate transfers ownership only at the stair-side lobby.
                if (self.stair_handoff_on_corridor_exit and
                        (all_rooms_exited or terminal_return_active)):
                    self._announce_stair_return_transit(
                        "four_rooms_exited_returning_to_truth_stair_gate"
                        if complete_room_direct_return else
                        "best_effort_supplement_returning_to_truth_stair_gate")
                self.corridor_sweep_history.append({
                    "event": ("ALL_ROOMS_EXITED_DIRECT_STAIR_RETURN"
                              if complete_room_direct_return else
                              "TERMINAL_RETURN_PASS_ACTIVE"),
                    "elapsed_sec": round(self.elapsed(), 3),
                    "exited_rooms": exited_rooms,
                    "target": (list(self.corridor_station_origin[:2])
                               if self.corridor_station_origin is not None
                               else None),
                })
            # Once raw corridor geometry is confirmed once after a short local
            # motion, allow only a forward transit probe before the full
            # latch. Door and
            # room takeover still require corridor_established, so a lobby
            # opening cannot pull the robot backward.
            raw_corridor_transit = bool(
                gate_context and gate_context.get("raw_is_corridor") and
                int(gate_context.get("confirmation_count", 0)) >= 1 and
                gate_pose is not None and self.mission_start_pose is not None and
                math.hypot(float(gate_pose[0]) -
                           float(self.mission_start_pose[0]),
                           float(gate_pose[1]) -
                           float(self.mission_start_pose[1])) >= 0.75)
            corridor_forward_lock = bool(
                (self.corridor_established or raw_corridor_transit or
                 self.corridor_entry_forward_lock) and
                not rooms_complete and self.room_scheduler.active_door is None)
            if (returning_home and all_rooms_exited and
                    self._stair_wait_reached(scheduling_pose)):
                # [PORTED zip140s A:11236-11237] 记录首次经证实的走廊入口返回。
                if getattr(self, "_record_corridor_return", None) is not None:
                    self._record_corridor_return(
                        "stair_wait_zone_reached_in_scheduler",
                        scheduling_pose)
                # The online map has confirmed the corridor terminus and all
                # discovered rooms have been exited.  The initial map-ready
                # pose is the only online, layout-independent reference for
                # the downstairs/G1 waiting area.  Stop navigation here so a
                # later floor-transition controller can take ownership; do
                # not restart FUEL or search arbitrary lobby frontiers.
                self.termination_reason = "STAIR_WAIT_ZONE_REACHED"
                self._set_state("STAIR_WAIT_ZONE", "mission_start_g1_area")
                self.corridor_sweep_history.append({
                    "event": "STAIR_WAIT_ZONE_REACHED",
                    "elapsed_sec": round(self.elapsed(), 3),
                    "target": list(self._stair_wait_target()[:2]),
                    "tolerance_m": self.stair_wait_tolerance,
                    "reason": "corridor_end_confirmed_return_pass_complete",
                })
                break
            # Room completion is not an immediate stop/rewind.  First use the
            # already confirmed corridor axis to advance through any remaining
            # known-free corridor.  The bounded sweep prevents a loop and only
            # then hands control to the ordinary home-return planner.
            # Give a confirmed opposite-side door one short, safe recentering
            # attempt before any new forward corridor sweep can carry the
            # robot farther past its station.
            # Before the corridor door-search phase is armed, a remembered
            # opposite-side candidate is necessarily behind the just-entered
            # corridor station.  Do not let it pull the robot into the lobby;
            # the normal paired-door behavior resumes after forward progress
            # has armed the search phase.
            goal = (self._plan_pending_paired_opposite_door()
                    if self.corridor_door_detection_armed else None)
            # [PORTED zip140s A:11263-11351] 出室后对面房间优先级: 已访问门
            # 证明了一个走廊站点; 任何新前进扫掠前, 要求先对该站点镜像侧做一次
            # 占据验证检查(把软加分变成实际配对房间约束, 仍拒绝穿越未观察墙)。
            _recent_exit_check = getattr(
                self, "_recent_exit_needs_opposite_check", None)
            _mirrored_plan = getattr(
                self, "_plan_mirrored_paired_room_entry_goal", None)
            if (goal is None and not all_rooms_exited and
                    self.room_scheduler.active_door is None and
                    _recent_exit_check is not None and
                    _recent_exit_check(gate_pose)):
                # Prefer a measured, confirmed A*/SCAN-safe portal already
                # visible at the exit station over a mirrored hypothesis.
                goal = self._plan_local_room_entry_goal()
                if goal is not None:
                    self.corridor_sweep_history.append({
                        "event": "POST_EXIT_MEASURED_OPPOSITE_SELECTED",
                        "elapsed_sec": round(self.elapsed(), 3),
                        "source": goal.get("source"),
                        "candidate_id": goal.get("local_door_candidate_id"),
                        "recent_exit_side": self.last_room_exit_side,
                        "policy": "measured_portal_before_mirrored_hypothesis",
                    })
                elif _mirrored_plan is not None:
                    goal = _mirrored_plan(allow_recent_exit=True)
                if goal is not None:
                    self.corridor_sweep_history.append({
                        "event": "POST_EXIT_OPPOSITE_ROOM_PRIORITY_SELECTED",
                        "elapsed_sec": round(self.elapsed(), 3),
                        "source": goal.get("source"),
                        "recent_exit_side": self.last_room_exit_side,
                        "policy": "same_station_opposite_before_forward",
                    })
                else:
                    # Consume the fresh *measured* doorway candidate in this
                    # same scheduling cycle before selecting any forward goal.
                    goal = self._plan_local_room_entry_goal()
                    if goal is not None:
                        self.corridor_sweep_history.append({
                            "event": "POST_EXIT_OPPOSITE_ROOM_RESCAN_SELECTED",
                            "elapsed_sec": round(self.elapsed(), 3),
                            "source": goal.get("source"),
                            "candidate_id": goal.get(
                                "local_door_candidate_id"),
                            "recent_exit_side": self.last_room_exit_side,
                            "policy": (
                                "fresh_measured_opposite_before_forward"),
                        })
                    elif (_recent_exit_check(gate_pose) and
                          self.elapsed() - self.last_local_rescan_completed >
                          0.75):
                        # Do not stack this 90-degree hold immediately after
                        # the mirrored planner has already completed a scan.
                        # Keep the body at this verified corridor station while
                        # the local raster/detector accumulates the opposite
                        # jamb; the recent-exit time window bounds this hold.
                        self.post_exit_opposite_station_holds = int(
                            getattr(self, "post_exit_opposite_station_holds",
                                    0)) + 1
                        self.corridor_sweep_history.append({
                            "event": "POST_EXIT_OPPOSITE_ROOM_STATION_HOLD",
                            "elapsed_sec": round(self.elapsed(), 3),
                            "recent_exit_side": self.last_room_exit_side,
                            "policy": "bounded_pair_confirmation_before_forward",
                            "hold_count": self.post_exit_opposite_station_holds,
                        })
                        self._perform_local_rescan(
                            angle_rad=0.5 * math.pi,
                            angular_speed=self.room_camera_sweep_speed,
                            reason="post_exit_opposite_room_station_hold",
                            visual_sweep=True, room_id=None)
                        continue
            # A continuity return is the bounded second observation pass for
            # missed rooms, but only a real ENTRY may interrupt the home goal.
            # A post-exit corridor_resume is redundant and must never become a
            # new outbound corridor sweep while the next staircase is waiting.
            return_room_time_available = (
                max(0.0, self.maximum_duration - self.elapsed()) >=
                self.corridor_partial_return_room_minimum_remaining)
            if (goal is None and terminal_return_active and
                    not all_rooms_exited and return_room_time_available):
                goal = self._plan_local_room_entry_goal()
                if (goal is not None and
                        goal.get("room_role") != "ENTRY"):
                    skipped_source = str(goal.get("source", ""))
                    if skipped_source == "corridor_resume_centerline":
                        self.room_scheduler.abandon_corridor_resume(
                            self.elapsed(), "terminal_return_direct_handoff")
                    self.corridor_sweep_history.append({
                        "event": "RETURN_PASS_NON_ENTRY_GOAL_SKIPPED",
                        "elapsed_sec": round(self.elapsed(), 3),
                        "source": skipped_source,
                        "action": "continue_directly_to_corridor_entrance",
                    })
                    goal = None
                if goal is None:
                    goal = self._plan_known_unvisited_door_retry_goal()
                if goal is None:
                    _mirrored_plan_port = getattr(
                        self, "_plan_mirrored_paired_room_entry_goal", None)
                    if _mirrored_plan_port is not None:
                        goal = _mirrored_plan_port()
                _mirrored_station_recenter = bool(
                    goal is not None and
                    str(goal.get("source", "")) ==
                    "mirrored_pair_station_recenter")
                if (goal is not None and
                        goal.get("room_role") != "ENTRY" and
                        not _mirrored_station_recenter):
                    # [PORTED zip140s A:11415-11428] 连续性返回是错过房间的有界
                    # 二次观察, 只有真实 ENTRY 可打断回家目标; post-exit 的
                    # corridor_resume 冗余且绝不能变成新一轮出站扫掠。
                    _skipped_source = str(goal.get("source", ""))
                    if _skipped_source == "corridor_resume_centerline":
                        self.room_scheduler.abandon_corridor_resume(
                            self.elapsed(), "terminal_return_direct_handoff")
                    self.corridor_sweep_history.append({
                        "event": "RETURN_PASS_NON_ENTRY_GOAL_SKIPPED",
                        "elapsed_sec": round(self.elapsed(), 3),
                        "source": _skipped_source,
                        "action": "continue_directly_to_corridor_entrance",
                    })
                    goal = None
                if goal is not None:
                    goal["terminal_return_supplemental_room"] = True
                    self.corridor_sweep_history.append({
                        "event": "RETURN_PASS_MISSED_ROOM_SELECTED",
                        "elapsed_sec": round(self.elapsed(), 3),
                        "door_id": goal.get("estimated_door_id"),
                        "source": goal.get("source"),
                        "policy": "entry_camera_view_then_immediate_exit",
                    })
            if goal is None:
                goal = (self._plan_post_room_corridor_sweep_goal(cycle)
                        if rooms_complete and not returning_home else None)
            # Corridor acquisition precedes every generic recovery.  In
            # particular, region-stall backtracking is valid inside a room
            # but must not pull a newly started mission back into the lobby
            # while a direct forward corridor ray is available.
            if (goal is None and not corridor_forward_lock and
                    not self.corridor_established):
                goal = self._plan_corridor_acquisition_goal(
                    cycle, gate_pose, gate_grid, gate_context)
            if goal is None and not corridor_forward_lock:
                goal = self._plan_region_stall_backtrack(cycle)
            if goal is None:
                if returning_home:
                    # Four-room completion returns directly to the latched
                    # corridor entrance (G2).  A separate terminal-wall
                    # escape step only adds a backwards-exploration rejection
                    # and is unnecessary after the normal ROOM_EXIT has
                    # already restored the corridor centreline.
                    goal = (self._plan_corridor_exit_handoff_goal()
                            if self.stair_handoff_on_corridor_exit else
                            self._plan_mission_return_goal())
                elif rooms_complete:
                    # Keep searching the forward corridor; never fall back to
                    # the home goal merely because the bounded sweep exhausted
                    # its replanning attempts without seeing a wall.
                    goal = self._plan_corridor_no_frontier_recovery(cycle)
                    if goal is None:
                        self._append_termination_decision(
                            "LOCAL_RESCAN_POST_ROOM_CORRIDOR", {}, {
                                "reason": "corridor_end_wall_not_confirmed"})
                        self._perform_local_rescan()
                        continue
                elif corridor_forward_lock:
                    # A confirmed corridor is a one-way exploration context
                    # until its terminal wall or all required rooms are done.
                    # Do not use region-stall/backtrack goals that can send
                    # the robot into the lobby or a previously covered area.
                    goal = self._plan_corridor_door_goal()
                    if goal is None:
                        goal = self._plan_local_room_entry_goal()
                    if goal is None:
                        # A failed portal remains a registered landmark and
                        # therefore cannot be rediscovered as a "new" door.
                        # Retry it from the restored corridor centreline
                        # before committing another 7 m sweep.
                        goal = self._plan_known_unvisited_door_retry_goal()
                    if goal is None:
                        # The other doorway at the current corridor station
                        # is a persistent, geometry-validated side branch.
                        # Visit it before continuing to the next station;
                        # this remains distinct from a generic backwards
                        # frontier or a return-to-lobby command.
                        goal = self._remembered_unvisited_corridor_branch_goal(
                            cycle, scheduling_pose, scheduling_grid)
                    if goal is None:
                        goal = self._plan_corridor_no_frontier_recovery(cycle)
                        goal = self._plan_corridor_no_frontier_recovery(cycle)
                    if goal is None:
                        # [PORTED zip140s A:11503-11504] 上楼真值走廊探测: 上层
                        # 投影图可能只有着陆区+首段走廊, 普通 frontier/A* 无目标;
                        # 用有界真值轴段曝光下一片, 房间检测与后续导航仍在线。
                        _truth_probe = getattr(
                            self, "_plan_upper_floor_truth_corridor_probe",
                            None)
                        if _truth_probe is not None:
                            goal = _truth_probe(cycle)
                else:
                    goal = self._plan_room_goal()
            # A visual verification is a single safe standoff view inside an
            # already-entered room.  It is considered before selecting the
            # normal exit, never before corridor/door discovery.
            if (not returning_home and self.depth_breadth.entry_observed):
                visual_goal = self._plan_visual_verify_goal()
                if visual_goal is not None:
                    goal = visual_goal
            if goal is None and not returning_home:
                goal = self._plan_corridor_door_goal()
            if goal is None and not returning_home:
                goal = self._plan_local_room_entry_goal()
            if goal is not None:
                score, return_code = {
                    "source": goal.get("source"),
                    "room_role": goal.get("room_role"),
                }, 0
                self._set_state("PLAN_ROOM_GOAL", goal.get("room_role"))
            elif corridor_forward_lock:
                # Never fall through to an unconstrained FUEL goal here: a
                # temporary frontier absence is not permission to turn back.
                self._append_termination_decision(
                    "LOCAL_RESCAN_CORRIDOR_FORWARD_LOCK", {}, {
                        "reason": "no_forward_or_side_candidate",
                        "rooms_complete": False})
                terminal_wall = self._forward_corridor_wall_observed(
                    gate_pose, gate_grid, gate_axis,
                    maximum_distance=
                        self.terminal_missing_room_retrace_wall_distance)
                immediate_terminal_retrace = bool(
                    terminal_wall and
                    self._on_station_corridor_centerline(gate_pose) and
                    self.enable_terminal_missing_room_retrace and
                    self._terminal_missing_room_retrace_progress_met(
                        gate_pose) and
                    not self.terminal_missing_room_retrace_issued and
                    self.corridor_reversal_count < self.corridor_reversal_limit)
                if immediate_terminal_retrace:
                    # The terminal-axis door fallback above has already had
                    # first refusal.  If it found no safe portal, start the
                    # one allowed reverse observation pass immediately.  A
                    # stationary generic rescan cannot restore the shortened
                    # corridor aspect ratio and cost 4--5 seconds per run.
                    self.corridor_axis = -np.asarray(gate_axis, dtype=float)
                    self.corridor_reversed = True
                    self.corridor_reversal_count += 1
                    self.terminal_missing_room_retrace_issued = True
                    self.terminal_missing_room_no_frontier_streak = 0
                    self._record_missing_room_retrace_start(gate_pose)
                    self.corridor_sweep_history.append({
                        "event": "TERMINAL_MISSING_ROOM_RETRACE_STARTED",
                        "elapsed_sec": round(self.elapsed(), 3),
                        "reason": "forward_terminal_wall_no_safe_side_door",
                        "rooms_complete": False,
                        "axis": [float(self.corridor_axis[0]),
                                 float(self.corridor_axis[1])],
                    })
                    rescan_ok = False
                    goal = self._plan_corridor_no_frontier_recovery(cycle)
                else:
                    rescan_ok = self._perform_local_rescan()
                    # A rescan is a bounded observation action, not a terminal
                    # state. If it finds nothing, keep advancing on the online
                    # corridor axis instead of starting another zero-velocity
                    # rescan cycle.
                    goal = self._plan_corridor_no_frontier_recovery(cycle)
                if goal is None:
                    self.terminal_missing_room_no_frontier_streak += 1
                    # At a real terminal the former forward-only rule caused
                    # repeated rescans at the same pose until the time limit.
                    # A single reverse corridor pass gives an unseen side
                    # doorway (notably the early room-0 portal) a second,
                    # opposite-direction observation.  It remains inside the
                    # corridor lock, so it cannot select a generic lobby
                    # frontier or start F1 handoff prematurely.
                    if (self.enable_terminal_missing_room_retrace and
                            self._terminal_missing_room_retrace_progress_met(
                                gate_pose) and
                            not self.terminal_missing_room_retrace_issued and
                            self.corridor_reversal_count <
                            self.corridor_reversal_limit and
                            terminal_wall and
                            self._on_station_corridor_centerline(gate_pose)):
                        self.corridor_axis = -np.asarray(gate_axis, dtype=float)
                        self.corridor_reversed = True
                        self.corridor_reversal_count += 1
                        self.terminal_missing_room_retrace_issued = True
                        self.terminal_missing_room_no_frontier_streak = 0
                        self._record_missing_room_retrace_start(gate_pose)
                        self.corridor_sweep_history.append({
                            "event": "TERMINAL_MISSING_ROOM_RETRACE_STARTED",
                            "elapsed_sec": round(self.elapsed(), 3),
                            "reason": "forward_terminal_wall",
                            "rooms_complete": False,
                            "axis": [float(self.corridor_axis[0]),
                                     float(self.corridor_axis[1])],
                        })
                        goal = self._plan_corridor_no_frontier_recovery(cycle)
                    if goal is None:
                        continue
                else:
                    self.terminal_missing_room_no_frontier_streak = 0
                score, return_code = {
                    "source": goal.get("source"),
                    "recovery": "corridor_forward_after_rescan",
                    "rescan_ok": bool(rescan_ok),
                }, 0
            else:
                goal, score, return_code = self._plan_fuel_goal(cycle)
            if (goal is not None and
                    not str(goal.get("source", "")).startswith(
                        "lightweight_room_") and
                    goal.get("source") not in (
                           "mission_return_home",
                           "stair_corridor_exit_handoff", "stair_corridor_escape",
                        # The corridor-exit handoff intentionally traverses
                        # already visited corridor cells.  It is a return
                        # route, not an exploration candidate, so applying
                        # the global visited-component rejection here traps
                        # the robot at the terminal wall.
                        "stair_corridor_exit_handoff", "stair_corridor_escape")):
                point = goal.get("position") or []
                component = (
                    self._visited_component_for_corridor_goal(
                        point, scheduling_pose, scheduling_grid)
                    if len(point) >= 2 else {"matched": False})
                if component.get("matched"):
                    self.corridor_sweep_history.append({
                        "event":
                            "VISITED_MAP_COMPONENT_GLOBAL_GOAL_REJECTED",
                        "elapsed_sec": round(self.elapsed(), 3),
                        "rejected_goal": list(point),
                        "rejected_source": goal.get("source"),
                        "component_cells":
                            component.get("component_cells"),
                        "matched_branch_ids":
                            component.get("matched_branch_ids", []),
                    })
                    replacement = self._plan_corridor_no_frontier_recovery(
                        cycle)
                    if replacement is not None:
                        goal = replacement
                        score, return_code = {
                            "source": goal.get("source"),
                            "recovery": "visited_map_component_exclusion",
                        }, 0
                    else:
                        goal, return_code = None, 1
            if (goal is not None and
                    not str(goal.get("source", "")).startswith(
                        "lightweight_room_") and
                    goal.get("source") not in ("stair_corridor_exit_handoff",
                                                 "stair_corridor_escape")):
                point = goal.get("position") or []
                if (len(point) >= 2 and
                        self.room_scheduler.goal_in_visited_room(
                            (float(point[0]), float(point[1])))):
                    self.corridor_sweep_history.append({
                        "event": "VISITED_ROOM_GLOBAL_GOAL_REJECTED",
                        "elapsed_sec": round(self.elapsed(), 3),
                        "rejected_goal": list(point),
                        "rejected_source": goal.get("source"),
                    })
                    replacement = self._plan_corridor_no_frontier_recovery(cycle)
                    if replacement is not None:
                        goal = replacement
                        score, return_code = {
                            "source": goal.get("source"),
                            "recovery": "visited_room_exclusion",
                        }, 0
                    else:
                        goal, return_code = None, 1
            if return_code != 0 or not goal:
                goal = self._plan_region_exit_recovery(cycle)
                recovery_kind = "entered_region_safe_exit"
                if goal is None:
                    goal = self._plan_corridor_no_frontier_recovery(cycle)
                    recovery_kind = "no_frontier_corridor_sweep"
                if goal is None:
                    eligible, checks, evidence = self._termination_evidence()
                    if eligible:
                        self._append_termination_decision(
                            "MISSION_TERMINATE", checks, evidence)
                        self.termination_reason = "NO_VALID_FRONTIER_CONFIRMED"
                        self._set_state("NO_VALID_FRONTIER_CONFIRMED")
                        break
                    local_goal = self._plan_local_room_entry_goal()
                    if local_goal is not None:
                        self._append_termination_decision(
                            "CONTINUE_EXPLORATION_LOCAL_ROOM_ENTRY",
                            checks, evidence)
                        goal = local_goal
                        recovery_kind = "local_room_entry_after_frontier_exhaustion"
                    else:
                        self._append_termination_decision(
                            "LOCAL_RESCAN", checks, evidence)
                        ok = self._perform_local_rescan()
                        if not ok:
                            # R28: rescan 被拒/无结果(如 locomotion_ready=false
                            # 或 executor 忙)后 continue 会无限重入,形成 214s
                            # 死锁。失败有界:连续达到上限后 break,走统一终止
                            # 流程(返程/交接),不再空转。
                            self.failed_local_rescan_streak += 1
                            if (self.failed_local_rescan_streak >=
                                    self.failed_local_rescan_limit):
                                self._append_termination_decision(
                                    "LOCAL_RESCAN_FAILED_LIMIT",
                                    checks, evidence)
                                self.termination_reason = \
                                    "LOCAL_RESCAN_FAILED_LIMIT"
                                self._set_state(
                                    "LOCAL_RESCAN_FAILED_LIMIT")
                                break
                        else:
                            self.failed_local_rescan_streak = 0
                        # Re-enter the complete selection chain with the newly
                        # accumulated occupancy rather than fabricating a goal.
                        continue
                score, return_code = {
                    "source": goal.get("source"),
                    "recovery": recovery_kind,
                }, 0
            # A room ENTRY is not complete until the robot has safely returned
            # through the portal. Refuse a new commit near the hard mission
            # deadline instead of reproducing run27's enter-at-297.9 s trap.
            if (str(goal.get("room_role", "")) == "ENTRY" and
                    not self.room_scheduler.entry_confirmed and
                    self.maximum_duration - self.elapsed() <
                    self.room_mission_entry_minimum_remaining):
                rejected_door = goal.get("estimated_door_id")
                remaining = max(0.0, self.maximum_duration - self.elapsed())
                self.room_scheduler.abort_active_room(
                    self.elapsed(), "mission_exit_reserve_entry_rejected",
                    cooldown_seconds=self.room_mission_entry_minimum_remaining)
                self.corridor_sweep_history.append({
                    "event": "ROOM_ENTRY_REJECTED_MISSION_EXIT_RESERVE",
                    "elapsed_sec": round(self.elapsed(), 3),
                    "door_id": rejected_door,
                    "remaining_mission_sec": round(remaining, 3),
                    "required_remaining_sec":
                        self.room_mission_entry_minimum_remaining,
                })
                continue
            path_result = (goal.pop("_preplanned_path_result", None) or
                           self._plan_path(cycle, goal))
            goal, path_result = self._intercept_room_bound_path(
                cycle, goal, path_result)
            goal, path_result = self._apply_corridor_sweep(
                cycle, goal, path_result)
            replan_count = 0
            if path_result["success"]:
                path_result = self._refine_path(cycle, path_result, goal)
            if (not path_result["success"] and path_result.get("request_astar_replan")):
                replan_count = 1
                goal["_preserve_astar_geometry"] = True
                if goal.get("region_entry_return"):
                    # A 3-D SCAN-lite rejection can reveal obstacle evidence
                    # absent from the 2-D projection used for the first route.
                    # Retry once with normal body clearance before asking the
                    # global explorer for an alternative escape.
                    goal["planning_clearance_m"] = max(
                        self.clearance,
                        float(goal.get("planning_clearance_m", 0.0)))
                path_result = self._plan_path(cycle, goal)
                if path_result["success"]:
                    path_result = self._refine_path(cycle, path_result, goal)
            safe_prefix = list(path_result.get("scan_lite_safe_prefix") or [])
            if (not path_result.get("success") and
                    str(goal.get("source", "")) == "corridor_sweep" and
                    safe_prefix):
                safe_target = safe_prefix[-1]
                with self.lock:
                    current_pose = self.pose
                if (current_pose is not None and
                        math.hypot(float(safe_target[0]) - current_pose[0],
                                   float(safe_target[1]) - current_pose[1]) >
                        self.reached_tolerance):
                    # Every prefix waypoint was footprint-checked before the
                    # first colliding waypoint.  Stop there, then let the
                    # next cycle plan around the new obstacle evidence.
                    goal["position"] = [float(safe_target[0]),
                                        float(safe_target[1]),
                                        float(current_pose[2])]
                    goal["scan_lite_safe_prefix_recovery"] = True
                    path_result = {
                        "success": True,
                        "reason": "scan_lite_safe_prefix_recovery",
                        "path": [(float(current_pose[0]),
                                  float(current_pose[1]))] + safe_prefix,
                        "execution_waypoints": safe_prefix,
                        "refined_path": [(float(current_pose[0]),
                                          float(current_pose[1]))] + safe_prefix,
                        "refined_path_length": polyline_length(
                            [(float(current_pose[0]), float(current_pose[1]))] +
                            safe_prefix),
                    }
                    self.corridor_sweep_history.append({
                        "event": "CORRIDOR_SCAN_LITE_SAFE_PREFIX_SELECTED",
                        "elapsed_sec": round(self.elapsed(), 3),
                        "cycle": cycle,
                        "safe_target": [float(safe_target[0]),
                                        float(safe_target[1])],
                        "waypoint_count": len(safe_prefix),
                    })
            if (not path_result.get("success") and
                    str(goal.get("source", "")) == "corridor_sweep" and
                    path_result.get("scan_lite_start_collision")):
                # run28 stopped at x=20.64: a newly integrated voxel occupied
                # the live footprint, so the same forward target failed three
                # times without ever reaching Room 2/3.  The one-way corridor
                # lock excludes generic region backtracking, therefore handle
                # this local map-start trap before global failure accounting.
                recovery_goal = self._plan_corridor_start_collision_backtrack(
                    cycle)
                if recovery_goal is not None:
                    goal = recovery_goal
                    score, return_code = {
                        "source": "corridor_start_collision_backtrack",
                        "recovery": "scan_lite_start_collision",
                    }, 0
                    path_result = goal.pop("_preplanned_path_result")
            if (not path_result.get("success") and
                    path_result.get("scan_lite_start_collision") and
                    self.depth_breadth.entry_observed):
                # A path cannot be refined while its very first footprint is
                # colliding. Switch immediately to recently occupied poses;
                # waiting for executor no-progress is impossible because the
                # rejected path is never sent to the executor.
                self.region_stall_backtrack_pending = True
                recovery_goal = self._plan_region_stall_backtrack(cycle)
                if recovery_goal is not None:
                    goal = recovery_goal
                    score, return_code = {
                        "source": "region_stall_backtrack",
                        "recovery": "scan_lite_start_collision",
                    }, 0
                    path_result = goal.pop("_preplanned_path_result")
            with self.lock:
                start_pose = self.pose
                self.active_goal_id = cycle
                self.active_goal_distance = 0.0
                self.active_goal_last_pose = self.pose
            self.active_goal_pub.publish(Int32(data=cycle))
            execution_started = self.elapsed()
            if not path_result["success"]:
                result = {
                    "success": False,
                    "reason": path_result["reason"],
                    "scan_lite_reason":
                        path_result.get("scan_lite_reason"),
                    "scan_lite_start_collision":
                        path_result.get("scan_lite_start_collision", False),
                }
            else:
                self._publish_speed_for_goal(goal)
                result = self._execute_path(path_result, goal, cycle)
            execution_ended = self.elapsed()
            with self.lock:
                final_pose = self.pose
                actual_distance = self.active_goal_distance
                self.active_goal_id = -1
                self.active_waypoint_id = -1
                self.active_goal_last_pose = None
            self.active_goal_pub.publish(Int32(data=-1))
            self.active_waypoint_pub.publish(Int32(data=-1))
            # The asynchronous writer is flushed every sample in this mode;
            # allow its worker to commit the final sample before integration.
            time.sleep(0.05)
            persisted_actual_distance = self._trajectory_length_from_jsonl(cycle)
            goal_position = goal.get("position") or []
            straight = (math.hypot(float(goal_position[0]) - start_pose[0],
                                   float(goal_position[1]) - start_pose[1])
                        if start_pose and len(goal_position) >= 2 else None)
            final_error = (math.hypot(float(goal_position[0]) - final_pose[0],
                                      float(goal_position[1]) - final_pose[1])
                           if final_pose and len(goal_position) >= 2 else None)
            reported_success = bool(result.get("success"))
            reported_reason = result.get("reason")
            # A confirmed doorway is allowed to preempt an in-flight
            # corridor-transit goal.  The executor correctly reports its
            # cancellation, but this is a scheduler-directed handoff, not a
            # navigation failure.  Treat it as a completed handoff so it
            # cannot accumulate into GOAL_FAILURE before the doorway goal is
            # selected on the next planning cycle.
            if (reported_reason ==
                    "goal_cancelled_by_manager:confirmed_local_door_preemption"):
                result = dict(result)
                result["executor_reported_success"] = False
                result["executor_reason"] = reported_reason
                result["success"] = True
                result["reason"] = "path_preempted_for_local_door"
                result["effective_scheduler_handoff"] = True
                reported_success = True
                reported_reason = result["reason"]
            # Do not let map-frame goal completion alone clear an active room.
            # If the live occupancy window does not contain the established
            # corridor, report a bounded EXIT failure to the room scheduler;
            # its next attempt reverses the physically traversed ENTRY path.
            # Door detection remains disabled while that room is active, so
            # room-interior edges cannot become new doors.
            if (reported_success and goal.get("room_role") == "EXIT" and
                    (self.room_exit_require_raw_corridor_confirmation or
                     self.room_exit_require_station_corridor_confirmation)):
                exit_confirmed, exit_reason = \
                    self._room_exit_corridor_confirmed(
                        final_pose, self.grid, goal)
                if not exit_confirmed:
                    result = dict(result)
                    result["executor_reported_success"] = True
                    result["executor_reason"] = reported_reason
                    result["success"] = False
                    result["reason"] = exit_reason
                    reported_success = False
                    reported_reason = exit_reason
                    self.corridor_sweep_history.append({
                        "event": "ROOM_EXIT_CORRIDOR_CONFIRMATION_REJECTED",
                        "elapsed_sec": round(self.elapsed(), 3),
                        "room_id": goal.get("room_id"),
                        "door_id": goal.get("estimated_door_id"),
                        "final_pose": (list(final_pose[:3])
                                       if final_pose is not None else None),
                        "reason": exit_reason,
                        "action": "retain_active_room_and_reverse_entry_trace",
                    })
            if (not reported_success and
                    (reported_reason == "goal_unreachable_no_progress" or
                     result.get("scan_lite_start_collision")) and
                    self.depth_breadth.entry_observed and
                    self.region_stall_backtrack_attempts <
                    self.region_stall_backtrack_limit):
                self.region_stall_backtrack_pending = True
            # [PORTED zip140s A:11998-12062] 活动目标定位恢复: 注册丢失是
            # 基础设施中断而非房间 ENTRY/EXIT/viewpoint 结果。必须在 record_result
            # 之前解决, 以免估计器故障消耗语义重试或清除活动房间事务。
            if (not reported_success and reported_reason in (
                    "fastlio_registration_lost",
                    "goal_cancelled_by_manager:"
                    "localization_truth_recovery_hold")):
                _recovery_wait = getattr(
                    self, "_wait_for_active_goal_localization_recovery", None)
                if _recovery_wait is not None:
                    # A 的模块级辅助函数(A:203-217); B 没有, 内联为局部函数。
                    def _port_structured_executor_cancel(execution_result,
                                                         reason):
                        if not isinstance(execution_result, dict):
                            return None
                        executor_goal = execution_result.get("goal")
                        try:
                            payload = {
                                "goal_sequence": int(execution_result.get(
                                    "goal_sequence", -1)),
                                "goal_stamp": float(
                                    execution_result["goal_stamp"]),
                                "goal_x": float(executor_goal["x"]),
                                "goal_y": float(executor_goal["y"]),
                                "reason": str(reason),
                            }
                        except (KeyError, TypeError, ValueError,
                                OverflowError):
                            return None
                        if not all(math.isfinite(payload[key])
                                   for key in ("goal_stamp", "goal_x",
                                               "goal_y")):
                            return None
                        return payload

                    _recovered = _recovery_wait(result, cycle, goal)
                    if rospy.is_shutdown():
                        break
                    if _recovered:
                        with self.lock:
                            _recovered_pose = self.pose
                            _recovered_grid_update = self.grid_update_count
                        consecutive_failures = 0
                        self._set_state(
                            "REPLAN", "fastlio_truth_recovery_confirmed")
                        self.corridor_sweep_history.append({
                            "event":
                                "ACTIVE_GOAL_LOCALIZATION_RECOVERED_REPLAN",
                            "elapsed_sec": round(self.elapsed(), 3),
                            "goal_id": cycle,
                            "discarded_goal_source": goal.get("source"),
                            "discarded_room_role": goal.get("room_role"),
                            "fresh_pose": (
                                list(_recovered_pose[:3])
                                if _recovered_pose is not None else None),
                            "fresh_grid_update": int(_recovered_grid_update),
                            "action":
                                "discard_old_goal_and_path_replan_from_live_state",
                        })
                        continue
                    # 单一固定恢复期限已耗尽: 保留既有真值楼梯交接, 否则中止。
                    if (self.stair_handoff_on_corridor_exit and
                            self.room_scheduler.active_door is None and
                            self._stair_return_handoff_authorized()):
                        self.termination_reason = (
                            "STAIR_CORRIDOR_EXIT_HANDOFF_TRUTH_REQUESTED")
                        self.multi_floor_handoff_pending = True
                        self._set_state(
                            "STAIR_LOBBY_HANDOFF", "fastlio_lost_truth_fallback")
                        self.corridor_sweep_history.append({
                            "event": "STAIR_TRUTH_FALLBACK_AFTER_FASTLIO_LOSS",
                            "elapsed_sec": round(self.elapsed(), 3),
                            "goal_id": cycle,
                            "reason": reported_reason,
                            "policy":
                                "truth_handoff_after_recovery_deadline_exhausted",
                        })
                    else:
                        self.termination_reason = "FASTLIO_LOCALIZATION_LOST"
                        self.corridor_sweep_history.append({
                            "event": "MISSION_ABORT_FASTLIO_LOCALIZATION_LOST",
                            "elapsed_sec": round(self.elapsed(), 3),
                            "goal_id": cycle,
                            "reason": reported_reason,
                            "action":
                                "bounded_truth_recovery_exhausted_then_finalize",
                        })
                    _terminal_cancel = _port_structured_executor_cancel(
                        result, "fastlio_localization_lost_mission_abort")
                    if _terminal_cancel is not None:
                        self.cancel_goal_pub.publish(String(
                            data=json.dumps(_terminal_cancel, sort_keys=True)))
                    break
            goal["execution_result"] = {
                "reason": reported_reason,
                "waypoints": result.get("waypoints"),
                "completed_waypoints": result.get("completed_waypoints"),
                "progressive_timeout": result.get("progressive_timeout"),
                "progress_timeout_sec": result.get("progress_timeout_sec"),
            }
            semantic_result = self.room_scheduler.record_result(
                goal, reported_success,
                ((final_pose[0], final_pose[1]) if final_pose else None),
                execution_ended,
                None if reported_success else reported_reason)
            success = bool(semantic_result.get("success", reported_success))
            if (success and goal.get("terminal_return_supplemental_room") and
                    goal.get("room_role") == "ENTRY"):
                immediate_exit = self.room_scheduler.request_immediate_exit(
                    execution_ended,
                    "return_pass_entry_observation_complete")
                self.corridor_sweep_history.append({
                    "event": "RETURN_PASS_ROOM_IMMEDIATE_EXIT_ARMED",
                    "elapsed_sec": round(self.elapsed(), 3),
                    "door_id": goal.get("estimated_door_id"),
                    "room_id": goal.get("room_id"),
                    "armed": bool(immediate_exit),
                })
            if success != reported_success:
                result = dict(result)
                result["executor_reported_success"] = reported_success
                result["executor_reason"] = reported_reason
                result["success"] = success
                result["reason"] = semantic_result.get("reason")
                result["effective_semantic_success"] = success
            # Capture the first successful non-room motion as an online G1
            # staging anchor.  It is a footprint-validated pose on the
            # actually traversed route, so returning here gives the stair
            # controller a stable handoff point without consulting layout or
            # Gazebo truth data.
            source = str(goal.get("source", ""))
            if (success and self.stair_wait_anchor is None and
                    final_pose is not None and self.mission_start_pose is not None and
                    not source.startswith("lightweight_room") and
                    source not in ("mission_return_home", "local_doorway_detector") and
                    math.hypot(float(final_pose[0]) - float(self.mission_start_pose[0]),
                               float(final_pose[1]) - float(self.mission_start_pose[1])) >=
                    self.stair_wait_tolerance):
                self.stair_wait_anchor = (float(final_pose[0]),
                                          float(final_pose[1]),
                                          float(final_pose[2]))
                self.corridor_sweep_history.append({
                    "event": "STAIR_WAIT_ANCHOR_CAPTURED",
                    "elapsed_sec": round(self.elapsed(), 3),
                    "anchor": list(self.stair_wait_anchor[:2]),
                    "source": source,
                })
            # Couple exploration and vision at the room viewpoints that are
            # already on the LiDAR route.  A short turn is requested only if
            # RGB-D still has a valid-depth coverage gap; it never adds a new
            # translational goal or a fixed 360-degree delay per room.
            role = goal.get("room_role")
            room_id = str(goal.get("room_id") or "")
            with self.lock:
                visual_status = dict(self.visual_coverage_status.get(room_id, {}))
            sweep_count = self.room_visual_sweep_counts.get(room_id, 0)
            visual_deepening = bool(getattr(
                self.room_scheduler, "visual_deepening_requested", False))
            # ENTRY is only a safe portal crossing. Rotate RGB-D only after
            # reaching the map-validated deep G1 observation point; a sweep
            # at the doorway cannot cover the room and steals G1 travel time.
            is_initial_view = role == "G1"
            is_occlusion_view = role == "G2"
            is_deep_visual_view = role in ("G3", "G4") and visual_deepening
            is_two_pose_lateral_view = bool(
                self.room_visual_two_pose_full_sweep_enabled and
                room_id in self.room_visual_two_pose_requested and
                role in ("G3", "G4") and sweep_count < 2)
            lidar_visual_cue = (self._lidar_visual_cue(room_id)
                                 if success and is_initial_view and room_id
                                 else None)
            needs_initial_sweep = (is_initial_view and
                                   not self.room_visual_two_pose_full_sweep_enabled and
                                   (self.room_visual_force_g1_fan or
                                    visual_status.get("visual_sweep_needed", False) or
                                    lidar_visual_cue is not None))
            # Some narrow/early-confirmed portals transition directly from
            # ENTRY to G3/G4 and legitimately skip a separate G1 goal.  Do
            # not lose the single cheap RGB-D fan in that case; use the first
            # successful normal room viewpoint rather than adding movement.
            needs_deferred_initial_sweep = (
                role in ("G3", "G4") and sweep_count == 0 and
                visual_status.get("visual_sweep_needed", False))
            if success and is_initial_view:
                self.corridor_sweep_history.append({
                    "event": "ROOM_RGBD_G1_SWEEP_GATE",
                    "elapsed_sec": round(self.elapsed(), 3),
                    "room_id": room_id, "source": goal.get("source"),
                    "camera_sweep_enabled": self.room_camera_sweep_enabled,
                    "force_g1_fan": self.room_visual_force_g1_fan,
                    "needs_initial_sweep": needs_initial_sweep,
                    "sweep_count": sweep_count,
                    "max_sweeps": self.room_camera_sweep_max_per_room,
                    "visual_sweep_needed": visual_status.get(
                        "visual_sweep_needed"),
                })
            if (success and self.room_camera_sweep_enabled and
                    (needs_initial_sweep or needs_deferred_initial_sweep or
                     is_occlusion_view or
                     is_deep_visual_view or is_two_pose_lateral_view) and room_id and
                    str(goal.get("source", "")).startswith("lightweight_room_") and
                    sweep_count < self.room_camera_sweep_max_per_room):
                angle = float(visual_status.get(
                    "suggested_sweep_angle_rad", self.room_camera_sweep_angle))
                if (self.room_visual_two_pose_full_sweep_enabled and
                        is_two_pose_lateral_view):
                    # A full revolution is a visual-recall diagnostic, not a
                    # semantic assertion: the RGB-D detector still decides
                    # whether a compact obstacle is red and sphere-like.
                    angle = (self.room_visual_full_sweep_angle
                             if sweep_count == 0 else
                             self.room_visual_secondary_sweep_angle)
                    direction = 1.0
                elif is_deep_visual_view:
                    # The safe G3/G4 point was selected from current LiDAR
                    # debt.  A compact fan there adds a genuinely different
                    # RGB-D baseline without a full room rotation.
                    angle = min(angle, 0.65)
                elif is_occlusion_view:
                    # G2 exists only when online LiDAR coverage reports a
                    # real obstacle shadow.  A short fan at this translated
                    # camera centre supplies the parallax that an in-place
                    # 360-degree turn cannot provide.
                    angle = self.room_visual_occlusion_sweep_angle
                elif is_initial_view:
                    # One route-preserving fan must span more than a single
                    # camera FOV; otherwise a room can be LiDAR-covered while
                    # most of its RGB-D inward hemisphere remains unseen.
                    angle = max(angle, self.room_visual_primary_min_angle)
                if (lidar_visual_cue is not None and
                        not self.room_visual_two_pose_full_sweep_enabled):
                    # A compact LiDAR object provides a concrete bearing.
                    # Sweep toward it with one camera half-FOV of margin so
                    # an imperfect FAST-LIO yaw estimate cannot put it just
                    # outside the RGB image.  This only changes direction,
                    # not the room route or the number of translational goals.
                    cue_bearing = float(lidar_visual_cue["bearing_rad"])
                    direction = 1.0 if cue_bearing >= 0.0 else -1.0
                    angle = max(angle, abs(cue_bearing) + math.radians(32.0))
                angle = min(self.room_camera_sweep_angle, max(math.pi / 6.0, angle))
                if (lidar_visual_cue is None or
                        self.room_visual_two_pose_full_sweep_enabled):
                    direction = float(visual_status.get(
                        "suggested_sweep_direction", 1.0))
                # [PORTED zip140s A:12304-12337] 双位姿横向扫掠以门内向法线
                # 为中心: 让两个 fan 的中点对准 door inward normal, 覆盖房间朝向
                # 半球而不增加请求角度/执行时间。helper(A:7940-7982, 无 self 的
                # 静态几何函数)内联为局部函数。
                def _port_sweep_direction_centered_on_target(
                        pose, target, angle_rad, fallback=1.0):
                    try:
                        if pose is None or target is None or len(target) < 2:
                            raise ValueError("missing sweep alignment geometry")
                        start_yaw = float(pose[3])
                        target_x = float(target[0])
                        target_y = float(target[1])
                        half_angle = 0.5 * abs(float(angle_rad))
                    except (IndexError, TypeError, ValueError):
                        return -1.0 if float(fallback) < 0.0 else 1.0
                    desired = math.atan2(target_y - float(pose[1]),
                                         target_x - float(pose[0]))
                    initial_error = math.atan2(
                        math.sin(desired - start_yaw),
                        math.cos(desired - start_yaw))
                    if abs(initial_error) <= math.radians(25.0):
                        if abs(initial_error) <= math.radians(2.0):
                            return -1.0 if float(fallback) < 0.0 else 1.0
                        return -1.0 if initial_error > 0.0 else 1.0

                    def midpoint_error(direction):
                        midpoint = start_yaw + direction * half_angle
                        return abs(math.atan2(
                            math.sin(desired - midpoint),
                            math.cos(desired - midpoint)))

                    positive_error = midpoint_error(1.0)
                    negative_error = midpoint_error(-1.0)
                    fallback_direction = -1.0 if float(fallback) < 0.0 else 1.0
                    if abs(positive_error - negative_error) <= 0.05:
                        return fallback_direction
                    return -1.0 if negative_error < positive_error else 1.0

                _sweep_alignment_target = None
                _sweep_alignment_policy = None
                if (self.room_visual_two_pose_full_sweep_enabled and
                        is_two_pose_lateral_view):
                    _active_door = self.room_scheduler.active_door
                    with self.lock:
                        _sweep_start_pose = self.pose
                    if _active_door is not None and _sweep_start_pose is not None:
                        _inward_x, _inward_y = _active_door.normal
                        _sweep_alignment_target = (
                            float(_sweep_start_pose[0]) + float(_inward_x),
                            float(_sweep_start_pose[1]) + float(_inward_y))
                        _sweep_alignment_policy = "room_inward_fan_midpoint"
                if (_sweep_alignment_target is not None and
                        isinstance(_sweep_alignment_target, (list, tuple)) and
                        len(_sweep_alignment_target) >= 2):
                    with self.lock:
                        _sweep_start_pose = self.pose
                    _fallback_direction = direction
                    direction = _port_sweep_direction_centered_on_target(
                        _sweep_start_pose, _sweep_alignment_target, angle,
                        fallback=direction)
                    self.corridor_sweep_history.append({
                        "event": "ROOM_VISUAL_SWEEP_INWARD_FAN_ALIGNED",
                        "elapsed_sec": round(self.elapsed(), 3),
                        "room_id": room_id,
                        "room_role": role,
                        "target": [float(_sweep_alignment_target[0]),
                                   float(_sweep_alignment_target[1])],
                        "fallback_direction": _fallback_direction,
                        "selected_direction": direction,
                        "policy": _sweep_alignment_policy,
                    })
                self.room_visual_sweep_counts[room_id] = sweep_count + 1
                self.corridor_sweep_history.append({
                    "event": "ROOM_RGBD_COVERAGE_GAP_SHORT_SWEEP",
                    "elapsed_sec": round(self.elapsed(), 3),
                    "room_id": room_id, "room_role": role,
                    "angle_rad": round(angle, 3), "direction": direction,
                    "deferred_initial_view": bool(needs_deferred_initial_sweep),
                    "coverage_before": visual_status.get("visual_coverage"),
                    "lidar_visual_cue": lidar_visual_cue,
                })
                self._perform_local_rescan(
                    angle_rad=angle,
                    angular_speed=self.room_camera_sweep_speed,
                    reason="room_rgbd_coverage_gap_short_sweep",
                    visual_sweep=True, room_id=room_id, direction=direction)
                # A strict candidate with only one or two views is stronger
                # evidence than a generic red pixel, yet it normally misses
                # the three-view tracker gate when the robot immediately
                # leaves this LiDAR waypoint.  Continue the *same* in-place
                # fan by a small angle so the detector gets a third parallax
                # sample.  This is route preserving and is deliberately
                # disabled for profile-relaxed blobs/red boxes.
                rospy.sleep(0.20)
                with self.lock:
                    candidate_status = dict(self.hazard_candidate_status.get(
                        room_id, {}))
                pending_strict = [item for item in candidate_status.get(
                    "pending_candidates", []) if
                    int(item.get("strict_hits", 0) or 0) in (1, 2) and
                    int(item.get("profile_relaxed_hits", 0) or 0) == 0 and
                    float(item.get("confidence", 0.0) or 0.0) >= 0.90]
                confirmation_nudge_angle = 0.45
                confirmation_nudge_duration = (confirmation_nudge_angle / max(
                    .05, self.room_camera_sweep_speed))
                if (self.room_visual_pending_strict_nudge_enabled and
                        pending_strict and sweep_count == 0 and
                        self.room_visual_sweep_counts.get(room_id, 0) <
                        self.room_camera_sweep_max_per_room and
                        self.maximum_duration - self.elapsed() >=
                        confirmation_nudge_duration +
                        self.empty_visual_supplement_minimum_remaining):
                    self.room_visual_sweep_counts[room_id] = (
                        self.room_visual_sweep_counts.get(room_id, 0) + 1)
                    self.corridor_sweep_history.append({
                        "event": "ROOM_RGBD_PENDING_STRICT_CONFIRMATION_NUDGE",
                        "elapsed_sec": round(self.elapsed(), 3),
                        "room_id": room_id, "room_role": role,
                        "angle_rad": confirmation_nudge_angle,
                        "direction": direction,
                        "candidate_ids": [item.get("id") for item in pending_strict],
                    })
                    self._perform_local_rescan(
                        angle_rad=confirmation_nudge_angle,
                        angular_speed=self.room_camera_sweep_speed,
                        reason="room_rgbd_pending_strict_confirmation",
                        visual_sweep=True, room_id=room_id, direction=direction)
                # A nearly complete visual fan with exactly one missing
                # sector is different from a broad coverage deficit: it is
                # best closed by a short turn in the mapper-suggested
                # direction, rather than by moving to another room waypoint.
                # This keeps the LiDAR route unchanged and targets only the
                # camera blind wedge that remains after the normal G1 fan.
                with self.lock:
                    completion_visual_status = dict(
                        self.visual_coverage_status.get(room_id, {}))
                completion_missing = list(completion_visual_status.get(
                    "missing_view_sectors", []) or [])
                completion_coverage = float(completion_visual_status.get(
                    "visual_coverage", 0.0) or 0.0)
                completion_angle = max(.35, min(1.0,
                    self.room_visual_one_sector_completion_angle))
                completion_duration = completion_angle / max(
                    .05, self.room_camera_sweep_speed)
                if (self.room_visual_one_sector_completion_enabled and
                        len(completion_missing) == 1 and .75 <= completion_coverage <
                        self.room_visual_coverage_target and
                        self.room_visual_sweep_counts.get(room_id, 0) <
                        self.room_camera_sweep_max_per_room and
                        self.maximum_duration - self.elapsed() >=
                        completion_duration +
                        self.empty_visual_supplement_minimum_remaining):
                    completion_direction = float(completion_visual_status.get(
                        "suggested_sweep_direction", direction))
                    self.room_visual_sweep_counts[room_id] = (
                        self.room_visual_sweep_counts.get(room_id, 0) + 1)
                    self.corridor_sweep_history.append({
                        "event": "ROOM_RGBD_ONE_SECTOR_COMPLETION_FAN",
                        "elapsed_sec": round(self.elapsed(), 3),
                        "room_id": room_id, "room_role": role,
                        "angle_rad": round(completion_angle, 3),
                        "direction": completion_direction,
                        "visual_coverage": round(completion_coverage, 3),
                        "missing_view_sector": completion_missing[0],
                    })
                    self._perform_local_rescan(
                        angle_rad=completion_angle,
                        angular_speed=self.room_camera_sweep_speed,
                        reason="room_rgbd_one_sector_completion",
                        visual_sweep=True, room_id=room_id,
                        direction=completion_direction)
                # A generic empty fan is not sufficient evidence for another
                # turn: it added a fixed delay in every room without improving
                # recall.  Only a sphere-like contour rejected *because* it
                # is clipped by a camera edge gets one directed recentering
                # turn. This uses online image evidence only.
                with self.lock:
                    red_status = dict(self.red_ball_search_status.get(
                        room_id, {}))
                    # The first fan has just completed, so use the refreshed
                    # coverage status rather than the value that selected it.
                    refreshed_visual_status = dict(
                        self.visual_coverage_status.get(room_id, {}))
                remaining = self.maximum_duration - self.elapsed()
                left_edge = int(red_status.get(
                    "left_edge_clipped_ball_frames", 0) or 0)
                right_edge = int(red_status.get(
                    "right_edge_clipped_ball_frames", 0) or 0)
                strict_before = int(red_status.get(
                    "strict_accepted_observations", 0) or 0)
                edge_evidence = (left_edge + right_edge) > 0
                # A second fan is reserved for a severe measured coverage
                # deficit (at least two unobserved camera sectors). This is
                # independent of whether a nearer red ball was already seen:
                # D9 remained hidden behind an unobserved sector in a room
                # that already contained other confirmed balls.
                no_red_evidence = int(red_status.get(
                    "red_pixel_frames", 0) or 0) == 0
                # The coverage callback can arrive one scheduling tick after
                # the local turn returns.  Retain the pre-turn debt as well,
                # otherwise an unseen room can incorrectly look resolved
                # merely because no fresh status has arrived yet.
                visual_coverage = min(
                    float(visual_status.get("visual_coverage", 1.0) or 0.0),
                    float(refreshed_visual_status.get(
                        "visual_coverage", 1.0) or 0.0))
                missing_sector_count = max(
                    len(visual_status.get("missing_view_sectors", []) or []),
                    len(refreshed_visual_status.get(
                        "missing_view_sectors", []) or []))
                coverage_debt_evidence = (
                    missing_sector_count >= 2 and
                    int(getattr(self.room_scheduler, "room_count", 0)) >=
                    self.room_visual_completion_minimum_rooms and
                    len(self.empty_visual_supplemented_rooms) <
                    self.room_visual_completion_room_budget and
                    (bool(visual_status.get("visual_sweep_needed", False)) or
                     bool(refreshed_visual_status.get(
                         "visual_sweep_needed", False))) and
                    visual_coverage <= 0.60)
                supplement_duration = (
                    self.empty_visual_supplement_angle /
                    max(.05, self.room_camera_sweep_speed))
                # Generic edge/empty-fan micro-sweeps are disabled for the
                # timed route.  A G3/G4 side view is different: it was already
                # selected from LiDAR coverage debt, so one edge recenter here
                # adds a real camera baseline and can bring a clipped D5-like
                # sphere into the central RGB-D field.
                if (self.empty_visual_supplement_enabled and
                        (edge_evidence or coverage_debt_evidence) and
                        room_id not in self.empty_visual_supplemented_rooms and
                        self.room_visual_sweep_counts.get(room_id, 0) <
                        self.room_camera_sweep_max_per_room and
                        remaining >= (supplement_duration +
                                      self.empty_visual_supplement_minimum_remaining) and
                        self.room_phase_target_seconds - self.elapsed() >=
                        supplement_duration + 5.0):
                    self.empty_visual_supplemented_rooms.add(room_id)
                    # A rejected red contour at an image edge identifies the
                    # useful turn direction.  Otherwise retain the ordinary
                    # opposite-fan policy.  This is strictly image evidence;
                    # no object location or map-truth is used here.
                    if coverage_debt_evidence:
                        supplement_direction = float(
                            refreshed_visual_status.get(
                                "suggested_sweep_direction", 1.0))
                    elif right_edge > left_edge:
                        supplement_direction = -1.0
                    elif left_edge > right_edge:
                        supplement_direction = 1.0
                    else:
                        supplement_direction = -1.0 if direction >= 0.0 else 1.0
                    self.corridor_sweep_history.append({
                        "event": ("ROOM_RGBD_COVERAGE_COMPLETION_FAN"
                                  if coverage_debt_evidence else
                                  "ROOM_RGBD_EDGE_CLIPPED_BALL_RECENTER"),
                        "elapsed_sec": round(self.elapsed(), 3),
                        "room_id": room_id,
                        "angle_rad": round(self.empty_visual_supplement_angle, 3),
                        "direction": supplement_direction,
                        "detector_frames": int(red_status.get("frames", 0) or 0),
                        "red_pixel_frames": int(red_status.get(
                            "red_pixel_frames", 0) or 0),
                        "left_edge_clipped_ball_frames": left_edge,
                        "right_edge_clipped_ball_frames": right_edge,
                        "visual_coverage": round(visual_coverage, 3),
                        "visual_coverage_target": self.room_visual_coverage_target,
                        "missing_sector_count": missing_sector_count,
                        "no_red_evidence": no_red_evidence,
                    })
                    self.room_visual_sweep_counts[room_id] = (
                        self.room_visual_sweep_counts.get(room_id, 0) + 1)
                    supplement_angle = (self.room_visual_completion_angle
                                        if coverage_debt_evidence else
                                        self.empty_visual_supplement_angle)
                    self._perform_local_rescan(
                        angle_rad=supplement_angle,
                        angular_speed=self.room_camera_sweep_speed,
                        reason=("room_rgbd_coverage_completion_fan"
                                if coverage_debt_evidence else
                                "room_rgbd_edge_clipped_ball_recenter"),
                        visual_sweep=True, room_id=room_id,
                        direction=supplement_direction)
                    # A coverage-debt fan can reveal its first useful red
                    # contour only in the final few frames.  In that precise
                    # case the ordinary two-turn budget would leave a genuine
                    # sphere clipped at the image edge (D5 in the diagnostic
                    # run) and therefore unconfirmable.  Re-read online
                    # detector status after the turn and allow one *shorter*
                    # reversal only for newly produced clipped-ball evidence.
                    if coverage_debt_evidence:
                        # Let the detector callback publish the final frames
                        # of the just-completed sweep before deciding whether
                        # a clipped contour needs a short reverse turn.
                        rospy.sleep(0.30)
                        with self.lock:
                            post_status = dict(self.red_ball_search_status.get(
                                room_id, {}))
                        post_left = int(post_status.get(
                            "left_edge_clipped_ball_frames", 0) or 0)
                        post_right = int(post_status.get(
                            "right_edge_clipped_ball_frames", 0) or 0)
                        new_edge_evidence = ((post_left > left_edge) or
                                             (post_right > right_edge))
                        followup_angle = min(.75,
                                             self.empty_visual_supplement_angle)
                        followup_duration = (followup_angle / max(
                            .05, self.room_camera_sweep_speed))
                        if (new_edge_evidence and
                                self.room_visual_sweep_counts.get(room_id, 0) <
                                self.room_camera_sweep_max_per_room and
                                self.maximum_duration - self.elapsed() >=
                                followup_duration +
                                self.empty_visual_supplement_minimum_remaining):
                            followup_direction = (-1.0 if post_right > post_left
                                                  else 1.0)
                            self.room_visual_sweep_counts[room_id] = (
                                self.room_visual_sweep_counts.get(room_id, 0) + 1)
                            self.corridor_sweep_history.append({
                                "event": "ROOM_RGBD_COVERAGE_DEBT_EDGE_FOLLOWUP",
                                "elapsed_sec": round(self.elapsed(), 3),
                                "room_id": room_id,
                                "angle_rad": round(followup_angle, 3),
                                "direction": followup_direction,
                                "new_left_edge_clipped_ball_frames":
                                    post_left - left_edge,
                                "new_right_edge_clipped_ball_frames":
                                    post_right - right_edge,
                            })
                            self._perform_local_rescan(
                                angle_rad=followup_angle,
                                angular_speed=self.room_camera_sweep_speed,
                                reason="room_rgbd_coverage_debt_edge_followup",
                                visual_sweep=True, room_id=room_id,
                                direction=followup_direction)
                    elif edge_evidence:
                        # A recentered strict sphere sometimes becomes fully
                        # visible only as the fan reaches its final angle. Give
                        # it one small reverse nudge to produce a third view
                        # with useful yaw parallax for the tracker.  This never
                        # runs for permissive/oblique red regions (the source
                        # of prior box false positives).
                        rospy.sleep(0.30)
                        with self.lock:
                            post_status = dict(self.red_ball_search_status.get(
                                room_id, {}))
                        strict_after = int(post_status.get(
                            "strict_accepted_observations", 0) or 0)
                        nudge_angle = 0.22
                        nudge_duration = nudge_angle / max(
                            .05, self.room_camera_sweep_speed)
                        if (strict_after > strict_before and
                                self.room_visual_sweep_counts.get(room_id, 0) <
                                self.room_camera_sweep_max_per_room and
                                self.maximum_duration - self.elapsed() >=
                                nudge_duration +
                                self.empty_visual_supplement_minimum_remaining):
                            nudge_direction = -supplement_direction
                            self.room_visual_sweep_counts[room_id] = (
                                self.room_visual_sweep_counts.get(room_id, 0) + 1)
                            self.corridor_sweep_history.append({
                                "event": "ROOM_RGBD_STRICT_CANDIDATE_PARALLAX_NUDGE",
                                "elapsed_sec": round(self.elapsed(), 3),
                                "room_id": room_id, "angle_rad": nudge_angle,
                                "direction": nudge_direction,
                                "new_strict_observations": strict_after - strict_before,
                            })
                            self._perform_local_rescan(
                                angle_rad=nudge_angle,
                                angular_speed=self.room_camera_sweep_speed,
                                reason="room_rgbd_strict_candidate_parallax_nudge",
                                visual_sweep=True, room_id=room_id,
                                direction=nudge_direction)
            if (success and role == "ENTRY" and room_id and
                    self.room_visual_two_pose_full_sweep_enabled and
                    room_id not in self.room_visual_two_pose_requested and
                    self.room_scheduler.request_visual_deepening(
                        room_id, "two_lateral_full_camera_coverage")):
                self.room_visual_two_pose_requested.add(room_id)
                self.corridor_sweep_history.append({
                    "event": "ROOM_RGBD_TWO_LATERAL_FULL_SWEEP_REQUESTED",
                    "elapsed_sec": round(self.elapsed(), 3),
                    "room_id": room_id,
                    "reference": "virtual_g1_center_no_spin",
                    "full_sweep_angle_rad": round(
                        self.room_visual_full_sweep_angle, 3),
                })
            if success and is_initial_view and room_id:
                with self.lock:
                    fresh_visual_status = dict(
                        self.visual_coverage_status.get(room_id, {}))
                    fresh_red_status = dict(self.red_ball_search_status.get(
                        room_id, {}))
                missing_sectors = list(fresh_visual_status.get(
                    "missing_view_sectors", []) or [])
                fresh_coverage = float(fresh_visual_status.get(
                    "visual_coverage", 1.0) or 0.0)
                completed_before_current = len(self.room_exit_map_snapshots)
                if (self.room_visual_two_pose_full_sweep_enabled and
                        room_id not in self.room_visual_two_pose_requested and
                        self.room_scheduler.request_visual_deepening(
                            room_id, "two_pose_full_camera_coverage")):
                    self.room_visual_two_pose_requested.add(room_id)
                    self.corridor_sweep_history.append({
                        "event": "ROOM_RGBD_TWO_POSE_FULL_SWEEP_REQUESTED",
                        "elapsed_sec": round(self.elapsed(), 3),
                        "room_id": room_id,
                        "first_pose_coverage": round(fresh_coverage, 3),
                        "full_sweep_angle_rad": round(
                            self.room_visual_full_sweep_angle, 3),
                    })
                elif (self.room_visual_deepening_enabled and missing_sectors and
                        len(self.room_visual_deepened_rooms) <
                        self.room_visual_deepening_budget and
                        completed_before_current >=
                        self.room_visual_deepening_min_completed_rooms and
                        fresh_coverage < self.room_visual_coverage_target and
                        self.maximum_duration - self.elapsed() >=
                        self.room_visual_deepening_min_remaining and
                        self.room_scheduler.request_visual_deepening(
                            room_id, "rgbd_no_red_coverage_debt")):
                    self.room_visual_deepened_rooms.add(room_id)
                    self.corridor_sweep_history.append({
                        "event": "ROOM_RGBD_DEEP_VIEW_REQUESTED",
                        "elapsed_sec": round(self.elapsed(), 3),
                        "room_id": room_id,
                        "missing_view_sectors": missing_sectors,
                        "visual_coverage": round(fresh_coverage, 3),
                        "remaining_deep_view_budget":
                            self.room_visual_deepening_budget -
                            len(self.room_visual_deepened_rooms),
                        "completed_rooms_before": completed_before_current,
                        "policy": "two_lidar_verified_g1_breadth_views",
                    })
                elif missing_sectors:
                    self.corridor_sweep_history.append({
                        "event": "ROOM_RGBD_DEEP_VIEW_NOT_REQUESTED",
                        "elapsed_sec": round(self.elapsed(), 3),
                        "room_id": room_id,
                        "missing_view_sectors": missing_sectors,
                        "visual_coverage": round(fresh_coverage, 3),
                        "remaining_deep_view_budget":
                            self.room_visual_deepening_budget -
                            len(self.room_visual_deepened_rooms),
                    })
            elif success and (is_deep_visual_view or is_two_pose_lateral_view) and room_id:
                if self.room_visual_two_pose_full_sweep_enabled:
                    count = self.room_visual_two_pose_completed_counts.get(
                        room_id, 0) + 1
                    self.room_visual_two_pose_completed_counts[room_id] = count
                    with self.lock:
                        two_pose_status = dict(self.visual_coverage_status.get(
                            room_id, {}))
                        two_pose_red_status = dict(self.red_ball_search_status.get(
                            room_id, {}))
                    visual_coverage = float(two_pose_status.get(
                        "visual_coverage", 0.0) or 0.0)
                    missing_sectors = list(two_pose_status.get(
                        "missing_view_sectors", []) or [])
                    clipped_ball_evidence = int(two_pose_red_status.get(
                        "left_edge_clipped_ball_frames", 0) or 0) + int(
                        two_pose_red_status.get(
                            "right_edge_clipped_ball_frames", 0) or 0)
                    strict_observations = int(two_pose_red_status.get(
                        "strict_accepted_observations", 0) or 0)
                    red_pixel_frames = int(two_pose_red_status.get(
                        "red_pixel_frames", 0) or 0)
                    # A clipped contour is a reason to obtain a second
                    # baseline only when the central full turn has not
                    # already supplied enough strict RGB-D observations for
                    # the tracker.  Previously any one edge frame forced a
                    # cross-room G4 even after dozens of valid observations,
                    # consuming 10--14 s and preventing the final room exit.
                    edge_recovery_needed = (
                        clipped_ball_evidence > 0 and
                        strict_observations < 3)
                    # Do not turn an *absence* of red pixels into a mandatory
                    # cross-room traversal.  A full G3 turn already samples
                    # every camera sector; when it sees no red evidence, the
                    # old rule sent the robot to G4 solely to search again.
                    # In room_3 this 2.3 m lateral transfer was the motion
                    # immediately preceding a FAST-LIO pose jump, so the
                    # robot stayed labelled "inside" a room after the map
                    # frame had become invalid.  A second view is justified
                    # only by incomplete visual coverage or an actual clipped
                    # red contour, both of which provide directional evidence.
                    unseen_red_recovery_needed = False
                    raw_second_view_needed = (
                        count < 2 and
                        (visual_coverage < self.room_visual_second_view_min_coverage or
                         len(missing_sectors) >= 2 or
                         edge_recovery_needed))
                    budget_available = (
                        room_id in self.room_visual_second_view_rooms or
                        len(self.room_visual_second_view_rooms) <
                        self.room_visual_second_view_budget)
                    needs_second_view = bool(raw_second_view_needed and
                                             budget_available)
                    if needs_second_view:
                        self.room_visual_second_view_rooms.add(room_id)
                    if count >= 2 or not needs_second_view:
                        self.room_scheduler.complete_virtual_visual_anchor(
                            room_id)
                        self.room_scheduler.clear_visual_deepening(
                            room_id,
                            ("two_lateral_full_camera_coverage_complete"
                             if count >= 2 else
                             "central_full_camera_coverage_sufficient"))
                    self.corridor_sweep_history.append({
                        "event": ("ROOM_RGBD_TWO_LATERAL_FULL_SWEEP_COMPLETE"
                                  if count >= 2 else
                                  ("ROOM_RGBD_SECOND_VIEW_REQUIRED" if
                                   needs_second_view else
                                   "ROOM_RGBD_CENTRAL_VIEW_SUFFICIENT")),
                        "elapsed_sec": round(self.elapsed(), 3),
                        "room_id": room_id,
                        "completed_lateral_views": count,
                        "visual_coverage": round(visual_coverage, 3),
                        "missing_view_sectors": missing_sectors,
                        "edge_clipped_ball_frames": clipped_ball_evidence,
                        "strict_accepted_observations": strict_observations,
                        "edge_recovery_needed": bool(edge_recovery_needed),
                        "unseen_red_recovery_needed": bool(
                            unseen_red_recovery_needed),
                        "second_view_budget": self.room_visual_second_view_budget,
                        "second_view_budget_available": bool(budget_available),
                        "second_view_required": bool(needs_second_view),
                        "full_sweep_angle_rad": round(
                            self.room_visual_full_sweep_angle, 3),
                    })
                else:
                    breadth_count = self.room_visual_breadth_counts.get(room_id, 0) + 1
                    self.room_visual_breadth_counts[room_id] = breadth_count
                    with self.lock:
                        breadth_status = dict(self.visual_coverage_status.get(
                            room_id, {}))
                    remaining_breadth = (breadth_count < self.room_visual_breadth_max_views and
                                         self.maximum_duration - self.elapsed() >=
                                         self.empty_visual_supplement_minimum_remaining)
                    if (not remaining_breadth or
                            float(breadth_status.get("visual_coverage", 0.0) or 0.0) >=
                            self.room_visual_coverage_target):
                        self.room_scheduler.clear_visual_deepening(
                            room_id, "g1_breadth_views_complete")
                    else:
                        self.corridor_sweep_history.append({
                            "event": "ROOM_RGBD_G1_BREADTH_OPPOSITE_VIEW_PENDING",
                            "elapsed_sec": round(self.elapsed(), 3),
                            "room_id": room_id,
                            "completed_breadth_views": breadth_count,
                            "max_breadth_views": self.room_visual_breadth_max_views,
                            "visual_coverage": round(float(breadth_status.get(
                                "visual_coverage", 0.0) or 0.0), 3),
                        })
            source = str(goal.get("source", ""))
            if source == "post_room_corridor_sweep":
                self.corridor_sweep_history.append({
                    "event": ("POST_ROOM_CORRIDOR_SWEEP_SUCCESS" if success else
                              "POST_ROOM_CORRIDOR_SWEEP_FAILED"),
                    "elapsed_sec": round(self.elapsed(), 3),
                    "goal": goal.get("position"),
                    "actual_distance_m": actual_distance,
                    "reason": reported_reason,
                })
                if not success:
                    self.post_room_corridor_sweep_done = True
            if (success and actual_distance >= 0.45 and source in (
                    "corridor_side_coverage_debt",
                    "corridor_side_branch_deepen",
                    "corridor_remembered_side_branch")):
                branch = self.branch_scheduler.find(
                    goal.get("corridor_branch_id"))
                if branch is not None:
                    self.branch_scheduler.mark_entered(branch.branch_id)
                    self.active_region_side_sign = float(branch.side)
                    self.corridor_sweep_history.append({
                        "event": "CORRIDOR_SIDE_BRANCH_COMMITTED_BY_EXECUTION",
                        "elapsed_sec": round(self.elapsed(), 3),
                        "branch_id": branch.branch_id,
                        "side": branch.side,
                        "goal_source": source,
                    })
            elif (success and actual_distance < 0.45 and source in (
                    "corridor_side_coverage_debt",
                    "corridor_side_branch_deepen",
                    "corridor_remembered_side_branch")):
                branch_id = goal.get("corridor_branch_id")
                branch_state = self.branch_scheduler.mark_entry_failed(
                    branch_id, self.elapsed())
                self.active_region_side_sign = None
                self.corridor_sweep_history.append({
                    "event": "CORRIDOR_SIDE_ZERO_PROGRESS_NOT_COMMITTED",
                    "elapsed_sec": round(self.elapsed(), 3),
                    "branch_id": branch_id,
                    "goal_source": source,
                    "actual_distance_m": actual_distance,
                    "branch_state": branch_state,
                })
            elif success and source == "side_region_corridor_return":
                branch_state = self.branch_scheduler.mark_exit(
                    self.elapsed(), coverage_complete=False)
                self.active_region_side_sign = None
                self.corridor_sweep_history.append({
                    "event": "CORRIDOR_SIDE_BRANCH_RETURN_CONFIRMED",
                    "elapsed_sec": round(self.elapsed(), 3),
                    "branch_state": branch_state,
                })
            elif (not success and source == "side_region_corridor_return"):
                # A return waypoint whose start footprint is blocked must not
                # be republished forever.  Cool down and release this branch,
                # allowing the corridor scheduler to advance to the next door
                # candidate instead of oscillating at the same side opening.
                branch_state = self.branch_scheduler.mark_exit(
                    self.elapsed(), coverage_complete=False, failed=True)
                self.active_region_side_sign = None
                self.corridor_sweep_history.append({
                    "event": "CORRIDOR_SIDE_BRANCH_RETURN_FAILED_COOLDOWN",
                    "elapsed_sec": round(self.elapsed(), 3),
                    "branch_state": branch_state,
                    "reason": reported_reason,
                })
            elif success and source.startswith("lightweight_room_exit"):
                # A physically confirmed exit is valuable visual evidence even
                # when the scheduler keeps the room marked incomplete (for
                # example, because its coverage-quality flag is conservative).
                # Figure 12 is an exit-history figure, so gate its snapshot on
                # the successful EXIT goal rather than on ``door.completed``.
                completed = bool(
                    self.room_scheduler.detector.doors and
                    self.room_scheduler.detector.doors[-1].completed)
                snapshot = self._capture_room_exit_map_snapshot(
                    goal.get("room_id"))
                # Save the side just exited for a short clearance window.
                # The opposite side is intentionally left fully eligible.
                with self.lock:
                    exit_pose = self.pose
                    exit_grid = self.grid
                exited_door = (self.room_scheduler.detector.doors[-1]
                               if self.room_scheduler.detector.doors else None)
                exit_context = (self._corridor_context(exit_pose, exit_grid)
                                if exit_pose is not None and exit_grid is not None
                                else None)
                if exited_door is not None and exit_context is not None:
                    axis = np.asarray(exit_context["axis"], dtype=float)
                    axis /= max(1e-9, float(np.linalg.norm(axis)))
                    normal = np.asarray([-axis[1], axis[0]], dtype=float)
                    # [PORTED zip140s A:6644-6662] _stable_corridor_axes: 局部
                    # 平行墙拟合的轴可能在扫描间翻转 pi, 翻转会连带翻转法线, 把
                    # 真实对面门变成同侧/后向候选。在线走廊站点坐标系建立后, 把
                    # 每次局部拟合对齐到它再分配门侧。纯插入对齐, 语义自包含。
                    _station_ref_axis = getattr(
                        self, "corridor_station_axis", None)
                    if _station_ref_axis is not None:
                        _ref_axis = np.asarray(_station_ref_axis, dtype=float)
                        _ref_axis /= max(1e-9, float(np.linalg.norm(_ref_axis)))
                        if float(np.dot(axis, _ref_axis)) < 0.0:
                            axis = -axis
                            normal = np.asarray([-axis[1], axis[0]],
                                                dtype=float)
                    center = np.asarray(exit_context["centerline_point"], dtype=float)
                    # ``corridor_side`` is intentionally close to the
                    # centreline and can cross its sign under a small
                    # FAST-LIO lateral correction.  That made a just-exited
                    # doorway look like the unvisited opposite side and
                    # repeatedly re-eligible at the same station.  The
                    # interior anchor is on the actual room side of the
                    # doorway and therefore gives a stable side label.
                    door_point = np.asarray(
                        getattr(exited_door, "interior_side", None) or
                        getattr(exited_door, "center", exit_pose[:2]), dtype=float)
                    self.last_room_exit_side = (1 if float(np.dot(
                        door_point[:2] - center, normal)) >= 0.0 else -1)
                    self.last_room_exit_pose = (float(exit_pose[0]),
                                                float(exit_pose[1]))
                    self.room_exit_resume_until = (
                        self.elapsed() + self.room_exit_resume_time)
                    # The first corridor station frame can be seeded from an
                    # oblique lobby/door observation.  Run45 retained a line
                    # 1.5 m beside the verified exit centre and consequently
                    # sent the next "forward" goal through room 1.  Rebase
                    # only the perpendicular component at every confirmed
                    # physical exit.  Longitudinal branch/door station values
                    # and the oriented axis are deliberately unchanged.
                    if (self.corridor_station_origin is not None and
                            self.corridor_station_axis is not None):
                        old_origin = np.asarray(
                            self.corridor_station_origin, dtype=float)
                        observed_center = np.asarray(center, dtype=float)
                        corrected = np.asarray(
                            rebase_corridor_line_laterally(
                                old_origin, self.corridor_station_axis,
                                observed_center), dtype=float)
                        correction = float(np.linalg.norm(
                            corrected - old_origin))
                        # An open doorway corrupts the observed centre: the
                        # lidar sees through the portal into the room interior
                        # and the "centreline" lands outside the physical
                        # corridor.  R18: the exit observation at x=-1.36
                        # shifted the station line 1.06 m into the west wall
                        # and every later return breadcrumb routed through the
                        # door, wedging the robot in the room entrance.  Never
                        # move the line by more than the membership tolerance
                        # - a shift that large puts the line outside the
                        # corridor band entirely, so keep the old line and the
                        # pre-rebase station behaviour (R17 success pattern).
                        max_correction = min(
                            self.corridor_exit_centerline_rebase_maximum,
                            self.corridor_centerline_membership_tolerance)
                        if correction <= max_correction:
                            self.corridor_station_origin = corrected
                            self.corridor_station_centerline_rebased = True
                            self.corridor_sweep_history.append({
                                "event":
                                    "CORRIDOR_STATION_CENTERLINE_REBASED_AT_EXIT",
                                "elapsed_sec": round(self.elapsed(), 3),
                                "lateral_correction_m": round(correction, 3),
                                "old_origin": old_origin.tolist(),
                                "new_origin": corrected.tolist(),
                                "observed_exit_center": observed_center.tolist(),
                                "axis_unchanged": True,
                            })
                        else:
                            self.corridor_sweep_history.append({
                                "event":
                                    "CORRIDOR_STATION_CENTERLINE_REBASE_REJECTED",
                                "elapsed_sec": round(self.elapsed(), 3),
                                "lateral_correction_m": round(correction, 3),
                                "maximum_m": round(max_correction, 3),
                                "old_origin": old_origin.tolist(),
                                "new_origin": corrected.tolist(),
                                "observed_exit_center":
                                    observed_center.tolist(),
                            })
                    # The final in-room visual side view can have a larger
                    # corridor station than the actual EXIT waypoint.  Keep
                    # that stale value and a real same-station opposite door
                    # is incorrectly labelled "behind progress" as soon as
                    # we return to the corridor.  An EXIT is a verified
                    # forward-progress boundary, so restart the local-door
                    # progress frame there.  Confirmed candidates already
                    # cached by the detector can then be preflighted in the
                    # very next cycle instead of driving a 7 m sweep first.
                    if self.corridor_station_axis is not None:
                        direction = (self.corridor_forward_station_sign
                                     if self.corridor_forward_station_sign
                                     is not None else 1.0)
                        self.corridor_forward_station_high_water = (
                            float(direction) * self._corridor_station(exit_pose))
                branch_state = self.branch_scheduler.mark_exit(
                    self.elapsed(), coverage_complete=completed)
                self.active_region_side_sign = None
                self.corridor_sweep_history.append({
                    "event": "CORRIDOR_SIDE_ROOM_EXIT_CONFIRMED",
                    "elapsed_sec": round(self.elapsed(), 3),
                    "branch_state": branch_state,
                    "coverage_complete": completed,
                    "recent_exit_side": self.last_room_exit_side,
                    "opposite_side_preserved": True,
                    "forward_high_water_reset_at_exit": (
                        self.corridor_forward_station_high_water),
                    "map_snapshot": snapshot,
                })
            self.goal_metrics.append({
                "goal_id": cycle, "start_time": execution_started,
                "end_time": execution_ended, "duration": execution_ended - execution_started,
                "start_pose": start_pose, "goal_pose": goal_position, "final_pose": final_pose,
                "straight_line_distance": straight,
                "astar_path_length": polyline_length(path_result.get("path", [])),
                "refined_path_length": path_result.get("refined_path_length"),
                "actual_trajectory_length": persisted_actual_distance,
                "actual_trajectory_length_source": "trajectory_timeseries.jsonl",
                "online_odometry_distance_diagnostic": actual_distance,
                "final_position_error": final_error, "final_yaw_error": None,
                "waypoint_count": result.get("waypoints", 0),
                "completed_waypoint_count": result.get("completed_waypoints", 0),
                "zero_command_stall_recovery_count": result.get(
                    "zero_command_stall_recovery_count", 0),
                "replan_count": replan_count, "success": bool(result.get("success")),
                "executor_reported_success": reported_success,
                "executor_reason": reported_reason,
                "failure_reason": None if result.get("success") else result.get("reason")})
            self.goal_success_count += int(success)
            self.goal_failure_count += int(not success)
            scheduler_phase = goal.get(
                "scheduler_phase", self.depth_breadth.phase(execution_started))
            scheduler_event = "goal_failed"
            if success and start_pose and final_pose:
                if goal.get("source") == "corridor_sweep":
                    travelled = (persisted_actual_distance
                                 if isinstance(persisted_actual_distance,
                                               (int, float))
                                 else actual_distance)
                    region_event = self.depth_breadth.record_success(
                        (start_pose[0], start_pose[1]),
                        (final_pose[0], final_pose[1]), travelled,
                        execution_ended)
                    scheduler_event = "corridor_forward_" + region_event
                elif str(goal.get("source", "")).startswith("lightweight_room_"):
                    scheduler_event = "room_semantic_goal_completed"
                elif goal.get("region_entry_return"):
                    scheduler_event = self.depth_breadth.record_entry_return()
                elif goal.get("region_stall_backtrack"):
                    scheduler_event = "region_stall_backtrack_completed"
                else:
                    travelled = (persisted_actual_distance
                                 if isinstance(persisted_actual_distance, (int, float))
                                 else actual_distance)
                    scheduler_event = self.depth_breadth.record_success(
                        (start_pose[0], start_pose[1]),
                        (final_pose[0], final_pose[1]), travelled,
                        execution_ended,
                        was_escape=bool(goal.get("depth_breadth_escape", False)))
            elif not success and goal.get("region_entry_return"):
                failed_target = goal.get("position") or []
                if len(failed_target) >= 2:
                    self.region_return_failed_targets.append((
                        float(failed_target[0]), float(failed_target[1])))
                scheduler_event = (
                    self.depth_breadth.record_entry_return_failed())
            scheduler_record = {
                "cycle": cycle,
                "elapsed_sec": round(self.elapsed(), 3),
                "phase": scheduler_phase,
                "event": scheduler_event,
                "goal_source": goal.get("source", "fuel_frontier"),
                "goal": goal.get("position"),
                "success": success,
                "failure_reason": None if success else result.get("reason"),
                "region_index": self.depth_breadth.region_index,
                "region_anchor": self.depth_breadth.anchor,
                "region_entry_anchor": self.depth_breadth.entry_anchor,
                "region_goal_count": self.depth_breadth.goal_count,
                "region_path_length": round(self.depth_breadth.path_length, 4),
                "escape_requested": self.depth_breadth.escape_requested,
                "escape_reason": self.depth_breadth.escape_reason,
                "region_coverage": self.depth_breadth.coverage_status,
                "entry_return_pending":
                    self.depth_breadth.entry_return_pending,
            }
            self.depth_breadth_history.append(scheduler_record)
            self.goal_history.append({
                "cycle": cycle, "elapsed_sec": round(self.elapsed(), 3),
                "goal": goal.get("position"), "information_gain": goal.get("information_gain"),
                "score": goal.get("score"), "source": goal.get("source", "fuel_frontier"),
                "frontier_center": goal.get("frontier_center"),
                "viewpoint_position": goal.get("viewpoint_position"),
                "planner_path_distance": goal.get("path_distance"),
                "execution_duplicate": goal.get("execution_duplicate"),
                "transit_revisit_fallback": goal.get(
                    "transit_revisit_fallback"),
                "room_id": goal.get("room_id"), "room_role": goal.get("room_role"),
                "estimated_door_id": goal.get("estimated_door_id"),
                "mandatory_portal_waypoints": goal.get("mandatory_portal_waypoints"),
                "room_goal_diagnostic": goal.get("room_goal_diagnostic"),
                "intercepted_goal": goal.get("intercepted_goal"),
                "intercepted_goal_source": goal.get("intercepted_goal_source"),
                "execution_timeout_sec": goal.get("execution_timeout_sec"),
                "progress_timeout_sec": goal.get("progress_timeout_sec"),
                "progressive_timeout": goal.get("progressive_timeout"),
                "exit_remaining_path_length_m":
                    goal.get("exit_remaining_path_length_m"),
                "exit_dynamic_timeout_sec":
                    goal.get("exit_dynamic_timeout_sec"),
                "exit_anchor_skipped": goal.get("exit_anchor_skipped"),
                "return_remaining_path_length_m":
                    goal.get("return_remaining_path_length_m"),
                "return_dynamic_timeout_sec":
                    goal.get("return_dynamic_timeout_sec"),
                "return_preflight_verified":
                    goal.get("return_preflight_verified"),
                "fallback_candidate": goal.get("fallback_candidate"),
                "breadcrumb_fallback": goal.get("breadcrumb_fallback"),
                "region_return_direct_astar":
                    goal.get("region_return_direct_astar"),
                "region_entry_original": goal.get("region_entry_original"),
                "region_entry_relocated":
                    goal.get("region_entry_relocated"),
                "region_return_relocation_distance_m":
                    goal.get("region_return_relocation_distance_m"),
                "region_return_search_radius_m":
                    goal.get("region_return_search_radius_m"),
                "region_return_candidates_tested":
                    goal.get("region_return_candidates_tested"),
                "region_exit_recovery_attempt":
                    goal.get("region_exit_recovery_attempt"),
                "planning_clearance_m": goal.get("planning_clearance_m"),
                "portal_preflight_verified": goal.get("portal_preflight_verified"),
                "scheduler_phase": scheduler_phase,
                "scheduler_event": scheduler_event,
                "success": success, "result": result,
            })
            self._set_state("GOAL_SUCCESS" if success else "GOAL_FAILURE",
                            result.get("reason"))
            if (reported_reason == "stair_truth_return_gate_reached" and
                    goal.get("source") == "stair_corridor_exit_handoff" and
                    self._stair_return_handoff_authorized()):
                # [PORTED zip140s A:13201-13202] GT 桥已确认物理楼梯侧大厅:
                # 记录真值闸门返回, 即使 FAST-LIO 地图位姿未达历史数值锚点。
                if getattr(self, "_record_corridor_return", None) is not None:
                    self._record_corridor_return(
                        "truth_stair_return_gate_reached", final_pose)
                # The GT bridge has confirmed the physical stair-side lobby.
                # End F1 ownership even if its accumulated FAST-LIO map pose
                # has not reached the historical numerical anchor.
                self.termination_reason = \
                    "STAIR_CORRIDOR_EXIT_HANDOFF_TRUTH_GATE"
                self._set_state(
                    "STAIR_LOBBY_HANDOFF", "truth_return_gate_reached")
                self.cancel_goal_pub.publish(String(
                    data="stair_truth_return_gate_handoff"))
                self.corridor_sweep_history.append({
                    "event": "STAIR_LOBBY_HANDOFF_TRUTH_GATE",
                    "elapsed_sec": round(self.elapsed(), 3),
                    "fastlio_pose": (list(final_pose[:3])
                                     if final_pose is not None else None),
                    "policy": "physical_gate_over_drifted_map_target",
                })
                break
            # A discontinuous FAST-LIO pose is already enough evidence that
            # the map frame is invalid.  Do not launch a local rescan: turning
            # in place cannot re-register an already jumped map and previously
            # produced hundreds of frozen-map updates before shutdown.
            # A portal EXIT can be physically verified by its final pose even
            # if FAST-LIO publishes one discontinuous sample while it is
            # crossing the doorway.  The scheduler has already accepted that
            # exit as successful above; aborting the whole mission here turns
            # a recoverable map-frame correction into loss of the remaining
            # rooms.  Keep the robot stopped briefly, then replan solely from
            # fresh live odometry/map data.  Other failures remain terminal.
            # ``localization_jump`` is a finite, single-frame map-frame
            # correction.  It can occur while entering a portal as well as
            # while leaving it.  The current goal/path is stale, but the next
            # planning cycle reads the settled FAST-LIO pose and latest grid;
            # aborting here unnecessarily loses all rooms beyond that portal.
            # Non-finite poses and lost registration remain terminal below.
            # A planar re-anchor is recoverable; a large vertical displacement
            # means the robot has fallen or the map/IMU state is invalid.  Do
            # not continue to create return goals at impossible Z values.
            recovered_pose_jump = bool(
                reported_reason == "localization_jump" and
                final_pose is not None and
                math.isfinite(float(final_pose[2])) and
                -0.80 <= float(final_pose[2]) <= 1.20)
            # A long return can finish physically at the corridor entrance
            # while FAST-LIO's vertical state becomes invalid on the final
            # few samples.  For the optional truth stair-entry test only,
            # hand off if the *horizontal* online pose is already at the
            # commanded entrance target.  A jump anywhere else remains a
            # hard stop; it must never be mistaken for a successful return.
            stair_return_near_target = bool(
                reported_reason == "localization_jump" and
                goal.get("source") == "stair_corridor_exit_handoff" and
                final_pose is not None and len(goal_position) >= 2 and
                math.isfinite(float(final_pose[0])) and
                math.isfinite(float(final_pose[1])) and
                math.hypot(float(final_pose[0]) - float(goal_position[0]),
                           float(final_pose[1]) - float(goal_position[1])) <= 1.20)
            if stair_return_near_target:
                # [PORTED zip140s A:13291-13293] 水平位姿已到受令入口目标的
                # 定位回退交接: 记录走廊返回。
                if getattr(self, "_record_corridor_return", None) is not None:
                    self._record_corridor_return(
                        "localization_fallback_return_target_xy_confirmed",
                        final_pose)
                self.termination_reason = "STAIR_LOBBY_HANDOFF_LOCALIZATION_FALLBACK"
                self._set_state("STAIR_LOBBY_HANDOFF", "return_target_xy_confirmed")
                self.corridor_sweep_history.append({
                    "event": "STAIR_LOBBY_HANDOFF_LOCALIZATION_FALLBACK",
                    "elapsed_sec": round(self.elapsed(), 3),
                    "target": list(goal_position[:2]),
                    "final_xy": [float(final_pose[0]), float(final_pose[1])],
                    "horizontal_error_m": round(math.hypot(
                        float(final_pose[0]) - float(goal_position[0]),
                        float(final_pose[1]) - float(goal_position[1])), 3),
                })
                break
            if recovered_pose_jump:
                self.corridor_sweep_history.append({
                    "event": "LOCALIZATION_JUMP_STOP_AND_REPLAN",
                    "elapsed_sec": round(self.elapsed(), 3),
                    "goal_id": cycle,
                    "room_role": goal.get("room_role"),
                    "goal_success_before_replan": bool(success),
                    "action": "stop_then_replan_from_fresh_fastlio_pose",
                })
                self.cancel_goal_pub.publish(String(
                    data="localization_jump_replan"))
                time.sleep(0.35)
            elif reported_reason in ("localization_jump", "localization_nonfinite",
                                     "localization_vertical_fault",
                                     "fastlio_registration_lost"):
                self.termination_reason = "FASTLIO_LOCALIZATION_LOST"
                self.corridor_sweep_history.append({
                    "event": "MISSION_ABORT_FASTLIO_LOCALIZATION_LOST",
                    "elapsed_sec": round(self.elapsed(), 3),
                    "goal_id": cycle,
                    "reason": reported_reason,
                    "action": "stop_motion_finalize_logs_and_visualization",
                })
                self.cancel_goal_pub.publish(String(
                    data="fastlio_localization_lost_mission_abort"))
                break
            if result.get("reason") == "TIME_LIMIT":
                self.termination_reason = "TIME_LIMIT"
                break
            if self.room_scheduler.state == "ROOM_EXIT_BLOCKED":
                self.termination_reason = "ROOM_EXIT_BLOCKED"
                break
            side_branch_source = str(goal.get("source", "")) in (
                "corridor_side_coverage_debt",
                "corridor_remembered_side_branch")
            room_entry_source = str(goal.get("source", "")) in (
                "lightweight_room_entry", "local_doorway_detector")
            corridor_resume_goal = (
                str(goal.get("source", "")) == "corridor_resume_centerline")
            paired_opposite_recenter_goal = (
                str(goal.get("source", "")) ==
                "paired_opposite_door_recenter")
            if not success and side_branch_source:
                branch_state = self.branch_scheduler.mark_entry_failed(
                    goal.get("corridor_branch_id"), self.elapsed())
                self.corridor_sweep_history.append({
                    "event": "CORRIDOR_SIDE_ENTRY_FAILED_COOLDOWN",
                    "elapsed_sec": round(self.elapsed(), 3),
                    "branch_id": goal.get("corridor_branch_id"),
                    "goal": goal.get("position"),
                    "reason": result.get("reason"),
                    "branch_state": branch_state,
                })
            if goal.get("source") == "stair_corridor_escape" and success:
                self.stair_return_escape_done = True
                self.corridor_sweep_history.append({
                    "event": "STAIR_RETURN_TERMINAL_ESCAPE_COMPLETE",
                    "elapsed_sec": round(self.elapsed(), 3),
                    "target": list(goal.get("position", [])[:2]),
                })
                continue
            if (goal.get("source") == "stair_corridor_exit_handoff" and success):
                if self._stair_return_handoff_authorized():
                    # [PORTED zip140s A:13424-13425] 走廊出口交接目标成功到达:
                    # 记录走廊返回后再终止本楼层。
                    if getattr(self, "_record_corridor_return", None) is not None:
                        self._record_corridor_return(
                            "corridor_exit_handoff_goal_reached", final_pose)
                    self.termination_reason = "STAIR_CORRIDOR_EXIT_HANDOFF"
                    self._set_state("STAIR_LOBBY_HANDOFF", "corridor_exit")
                    self.corridor_sweep_history.append({
                        "event": "STAIR_LOBBY_HANDOFF",
                        "elapsed_sec": round(self.elapsed(), 3),
                        "target": list(goal.get("position", [])[:2]),
                        "reason": ("rooms_complete_return_pass_complete"
                                   if self._mission_rooms_physically_exited()
                                   else
                                   "terminal_corridor_return_pass_complete"),
                    })
                    break
                # [PORTED zip140s A:13444-13445] G2 已到达但房间缺失:
                # 在下方 retry/终止决策前记录这次物理走廊返回。
                if getattr(self, "_record_corridor_return", None) is not None:
                    self._record_corridor_return(
                        "g2_return_goal_reached_with_missing_rooms",
                        final_pose)
                # G2 is not an upstairs handoff while rooms remain missing.
                # Finish safely instead of repeatedly reissuing a zero-length
                # return goal or allowing the stair node to take over early.
                # Retry policy: unlatch the terminal return up to the retry
                # limit so the corridor planner can resume exploration (fresh
                # doors off cooldown, reverse observation pass).  Progress in
                # the exited-room count resets the counter; repeated
                # no-progress G2 returns terminate as before.
                current_exited = sum(bool(door.visited) for door in
                                     self.room_scheduler.detector.doors)
                if current_exited > self._g2_last_exited_count:
                    self.g2_missing_rooms_retries = 0
                self._g2_last_exited_count = current_exited
                self.g2_missing_rooms_retries += 1
                if self.g2_missing_rooms_retries >= \
                        self.g2_missing_rooms_retry_limit:
                    self.termination_reason = "G2_REACHED_WITH_MISSING_ROOMS"
                    self._set_state("G2_MISSING_ROOMS",
                                    "return_pass_no_local_door")
                    self.corridor_sweep_history.append({
                        "event": "G2_RETURN_PASS_INCOMPLETE",
                        "elapsed_sec": round(self.elapsed(), 3),
                        "target": list(goal.get("position", [])[:2]),
                        "exited_rooms": current_exited,
                        "retry_attempts": self.g2_missing_rooms_retries,
                    })
                    break
                self.corridor_terminal_return_latched = False
                self.corridor_partial_return_trigger_reason = None
                self.corridor_sweep_history.append({
                    "event": "G2_MISSING_ROOMS_RETRY_EXPLORATION",
                    "elapsed_sec": round(self.elapsed(), 3),
                    "target": list(goal.get("position", [])[:2]),
                    "attempt": self.g2_missing_rooms_retries,
                    "limit": self.g2_missing_rooms_retry_limit,
                    "exited_rooms": current_exited,
                    "action": "terminal_return_unlatched_resume_exploration",
                })
            # [PORTED zip140s A:13470/13483-13504] 严格楼层交接闸: 终点墙
            # 反向补扫是恢复机制而非楼层完成证明。快返成功回调在缺房间时不得
            # 绕过严格四 EXIT 闸终止; 重新武装出站搜索继续探索。
            # _floor_handoff_room_requirement_met(A:6451-6460) 内联。
            _handoff_room_req_met = True
            if not getattr(self, "require_all_rooms_for_floor_handoff", True):
                _handoff_room_req_met = True
            else:
                _req_exits = max(4, int(self.room_target_count))
                _handoff_room_req_met = bool(
                    self.room_scheduler.active_door is None and
                    sum(bool(door.visited) for door in
                        self.room_scheduler.detector.doors) >= _req_exits)
            if (goal.get("source") == "mission_return_home" and success and
                    self._stair_wait_reached(final_pose) and
                    not _handoff_room_req_met):
                exited_now = sum(
                    bool(door.visited)
                    for door in self.room_scheduler.detector.doors)
                self.corridor_terminal_return_latched = False
                self.corridor_partial_return_trigger_reason = None
                self.corridor_sweep_history.append({
                    "event": "MISSION_RETURN_HOME_REJECTED_MISSING_ROOMS",
                    "elapsed_sec": round(self.elapsed(), 3),
                    "exited_rooms": exited_now,
                    "required_exits": max(4, int(self.room_target_count)),
                    "action": "rearm_outbound_corridor_room_search",
                })
                consecutive_failures = 0
                continue
            if (goal.get("source") == "mission_return_home" and success and
                    self._stair_wait_reached(final_pose)):
                self.termination_reason = "STAIR_WAIT_ZONE_REACHED"
                self._set_state("STAIR_WAIT_ZONE", "mission_start_g1_area")
                # [PORTED zip140s A:13471-13472] 快返目标到达: 记录走廊返回。
                if getattr(self, "_record_corridor_return", None) is not None:
                    self._record_corridor_return(
                        "mission_return_home_goal_reached", final_pose)
                self.corridor_sweep_history.append({
                    "event": "STAIR_WAIT_ZONE_REACHED",
                    "elapsed_sec": round(self.elapsed(), 3),
                    "target": list(self._stair_wait_target()[:2]),
                    "tolerance_m": self.stair_wait_tolerance,
                    "reason": "fast_return_goal_reached",
                })
                break
            consecutive_failures = 0 if success else consecutive_failures + 1
            exited_rooms = sum(bool(door.visited) for door in
                               self.room_scheduler.detector.doors)
            _probe_result_handler = getattr(
                self, "_handle_upper_floor_truth_probe_result", None)
            if _probe_result_handler is not None and _probe_result_handler(
                    success, source, reported_reason, exited_rooms):
                consecutive_failures = 0
            if not success and paired_opposite_recenter_goal:
                evidence = goal.get("paired_opposite_door_evidence") or {}
                identifier = str(evidence.get("candidate_id", "local-door"))
                self.pending_paired_opposite_door = None
                self.local_candidate_attempts[identifier] = self.elapsed()
                # This is a stale local anchor, not a mission failure.  Keep
                # moving toward the terminal wall so the LiDAR return latch
                # and any newly observed room doorway remain reachable.
                consecutive_failures = 0
                self.corridor_sweep_history.append({
                    "event": "PAIRED_OPPOSITE_RECENTER_UNREACHABLE_CONTINUE_FORWARD",
                    "elapsed_sec": round(self.elapsed(), 3),
                    "candidate_id": identifier,
                    "reason": reported_reason,
                    "cooldown_seconds": 20.0,
                })
            if side_branch_source:
                # Failure of one optional lateral endpoint is local evidence,
                # not failure of the exploration backend.  Continue forward
                # while the branch cooldown prevents immediate repetition.
                consecutive_failures = 0
            if room_entry_source and not success:
                # A failed entry is local doorway evidence, not a mission-wide
                # planner failure.  The door scheduler has already cooled the
                # candidate; keep traversing the corridor so another doorway
                # can be tested instead of terminating after three retries.
                consecutive_failures = 0
                portal_geometry_blocked = bool(
                    str(reported_reason).startswith("portal_segment_") and
                    (str(reported_reason).endswith("goal_footprint_blocked") or
                     str(reported_reason).endswith("no_path")))
                exited_rooms = sum(bool(door.visited) for door in
                                   self.room_scheduler.detector.doors)
                near_outbound_high_water = False
                if (final_pose is not None and
                        self.corridor_station_axis is not None and
                        self.corridor_forward_station_high_water is not None):
                    sign = (self.corridor_forward_station_sign
                            if self.corridor_forward_station_sign is not None
                            else 1.0)
                    directed = float(sign) * self._corridor_station(final_pose)
                    near_outbound_high_water = bool(
                        self.corridor_forward_station_high_water - directed <=
                        self.terminal_door_context_fallback_drawdown + 0.75)
                if (portal_geometry_blocked and near_outbound_high_water and
                        exited_rooms >= max(1, self.room_target_count - 2) and
                        not self.room_scheduler.entry_retry_pending()):
                    # This is a 3-D-confirmed blocked terminal-side portal,
                    # not a reason to continue through the apparent free
                    # pixels beyond the end wall.  Release it and start the
                    # one bounded reverse observation pass immediately so
                    # the opposite/far doorway can be considered.
                    failed_door = str(goal.get("estimated_door_id", ""))
                    self.room_scheduler.abort_active_room(
                        self.elapsed(), "terminal_portal_geometry_blocked",
                        cooldown_seconds=20.0)
                    if (not self.terminal_missing_room_retrace_issued and
                            self.corridor_reversal_count <
                            self.corridor_reversal_limit):
                        reverse_axis = np.asarray(
                            self.corridor_station_axis, dtype=float)
                        reverse_axis /= max(
                            1e-9, float(np.linalg.norm(reverse_axis)))
                        self.corridor_axis = -reverse_axis
                        self.corridor_reversed = True
                        self.corridor_reversal_count += 1
                        self.terminal_missing_room_retrace_issued = True
                        self.terminal_missing_room_no_frontier_streak = 0
                        self._record_missing_room_retrace_start(final_pose)
                        self.corridor_sweep_history.append({
                            "event":
                                "TERMINAL_BLOCKED_PORTAL_RETRACE_STARTED",
                            "elapsed_sec": round(self.elapsed(), 3),
                            "failed_door_id": failed_door,
                            "failure_reason": str(reported_reason),
                            "exited_rooms": exited_rooms,
                            "axis": [float(self.corridor_axis[0]),
                                     float(self.corridor_axis[1])],
                            "policy":
                                "no_forward_goal_beyond_terminal_failure",
                        })
                elif (portal_geometry_blocked and near_outbound_high_water and
                      self.room_scheduler.entry_retry_pending()):
                    # The nominal path can be invalidated by a newly inflated
                    # cell at the first narrow portal segment.  Do not let the
                    # terminal-wall guard erase the scheduler's already
                    # bounded second attempt: it uses a shorter staging path
                    # and is the recovery needed by the real floor-1 room-3
                    # doorway observed in run39.
                    self.corridor_sweep_history.append({
                        "event": "TERMINAL_PORTAL_ENTRY_RETRY_PRESERVED",
                        "elapsed_sec": round(self.elapsed(), 3),
                        "door_id": goal.get("estimated_door_id"),
                        "failure_reason": str(reported_reason),
                        "entry_attempts":
                            self.room_scheduler.entry_attempts,
                        "entry_retry_limit":
                            self.room_scheduler.config.entry_retry_limit,
                    })
                self.corridor_sweep_history.append({
                    "event": "ROOM_ENTRY_FAILURE_LOCAL_COOLDOWN",
                    "elapsed_sec": round(self.elapsed(), 3),
                    "door_id": goal.get("estimated_door_id"),
                    "goal": goal.get("position"),
                    "reason": reported_reason,
                })
                if reported_reason in ("semantic_goal_timeout",
                                       "room_budget_timeout",
                                       "goal_executor_timeout",
                                       "exit_progress_stalled",
                                       "goal_unreachable_no_progress",
                                       "scan_lite_refinement_failed"):
                    # A controller failure after motion is one-shot, but the
                    # first SCAN-lite failure happened before motion.  Keep
                    # that door active so LightweightRoomScheduler can use its
                    # footprint-verified shallow staging fallback on the next
                    # cycle.  Aborting here used to discard far Room 3 after
                    # ``disconnected_refined_segment`` even though the
                    # fallback was already armed by record_result().
                    keep_active_for_staging_retry = False
                    if reported_reason == "scan_lite_refinement_failed":
                        door_id = str(goal.get("estimated_door_id", ""))
                        failures = int(self.room_entry_refinement_failures.get(
                            door_id, 0)) + 1
                        self.room_entry_refinement_failures[door_id] = failures
                        if failures == 1:
                            keep_active_for_staging_retry = bool(
                                self.room_scheduler.active_door is not None)
                            self.corridor_sweep_history.append({
                                "event": "ROOM_ENTRY_REFINEMENT_STAGING_RETRY_ARMED",
                                "elapsed_sec": round(self.elapsed(), 3),
                                "door_id": door_id,
                                "retry_mode": "shallow_staging",
                                "cooldown_seconds": 0.0,
                            })
                    if not keep_active_for_staging_retry:
                        self.room_scheduler.abort_active_room(
                            self.elapsed(), "room_entry_failure_resume_corridor")
            if corridor_resume_goal:
                consecutive_failures = 0
                if success:
                    self.corridor_resume_failures = 0
                    self.room_scheduler.abandon_corridor_resume(
                        self.elapsed(), "corridor_recenter_succeeded")
                else:
                    self.corridor_resume_failures += 1
                    self.corridor_sweep_history.append({
                        "event": "CORRIDOR_RESUME_FAILED",
                        "elapsed_sec": round(self.elapsed(), 3),
                        "failure_count": self.corridor_resume_failures,
                        "failure_limit": self.corridor_resume_failure_limit,
                        "reason": result.get("reason"),
                        "goal": goal.get("position"),
                    })
                    if (self.corridor_resume_failures >=
                            self.corridor_resume_failure_limit):
                        self.room_scheduler.abandon_corridor_resume(
                            self.elapsed(), "corridor_recenter_failure_limit")
            if str(goal.get("source", "")).startswith("lightweight_room_"):
                consecutive_failures = 0
            if goal.get("corridor_start_collision_backtrack"):
                # This bounded escape is local recovery, never evidence that
                # the exploration backend has exhausted its global retries.
                consecutive_failures = 0
            if (success and source == "corridor_sweep" and
                    self.corridor_start_collision_backtrack_attempts > 0):
                self.corridor_sweep_history.append({
                    "event": "CORRIDOR_START_COLLISION_RECOVERY_RESET",
                    "elapsed_sec": round(self.elapsed(), 3),
                    "previous_attempts":
                        self.corridor_start_collision_backtrack_attempts,
                    "reason": "verified_forward_corridor_goal_succeeded",
                })
                self.corridor_start_collision_backtrack_attempts = 0
            if result.get("reason") in (
                    "door_candidate_unconfirmed", "door_candidate_rejected",
                    "confirmed_door_without_safe_g1"):
                consecutive_failures = 0
            if goal.get("region_entry_return"):
                # Exiting an incomplete branch is recovery work. Failure marks
                # the saved entry as bypassed so the next cycle can ask FUEL
                # for another escape; it must not terminate the whole mission.
                consecutive_failures = 0
            if goal.get("region_stall_backtrack"):
                consecutive_failures = 0
            if consecutive_failures >= self.maximum_consecutive_failures:
                self.termination_reason = "GOAL_FAILURE"
                break
            self._set_state("UPDATE_MAP")
            completed_room_entry = bool(
                success and goal.get("room_role") == "ENTRY" and
                str(goal.get("source", "")).startswith("lightweight_room_"))
            if completed_room_entry:
                # The preceding portal path was physically executed and
                # revalidated.  A direct occupancy callback is sufficient to
                # refresh the first in-room plan; polling the independently
                # written map-statistics file here used to add 3--7 s per
                # room without adding LiDAR coverage or a safety check.
                with self.lock:
                    previous_grid_update = int(self.grid_update_count)
                deadline = time.monotonic() + self.room_entry_replan_wait
                fresh_grid = False
                while not rospy.is_shutdown() and time.monotonic() < deadline:
                    with self.lock:
                        fresh_grid = self.grid_update_count > previous_grid_update
                    if fresh_grid:
                        break
                    time.sleep(0.05)
                self.corridor_sweep_history.append({
                    "event": "ROOM_ENTRY_FAST_REPLAN_WINDOW",
                    "elapsed_sec": round(self.elapsed(), 3),
                    "room_id": goal.get("room_id"),
                    "wait_limit_s": self.room_entry_replan_wait,
                    "fresh_grid_received": bool(fresh_grid),
                })
            else:
                # The in-memory grid is the source consumed by the next A*
                # and SCAN-lite pass.  Waiting on the asynchronously written
                # map-statistics JSON here caused repeated 3--8 s idle gaps
                # after otherwise successful corridor/room goals.  One new
                # grid callback is sufficient; the bounded timeout preserves
                # progress if the callback cadence is temporarily slow.
                with self.lock:
                    previous_grid_update = int(self.grid_update_count)
                deadline = time.monotonic() + self.replan_grid_wait
                while not rospy.is_shutdown() and time.monotonic() < deadline:
                    with self.lock:
                        fresh_grid = self.grid_update_count > previous_grid_update
                    if fresh_grid:
                        break
                    time.sleep(0.05)
            self._set_state("REPLAN")
        # [PORTED zip140s A:13971-14019] TIME_LIMIT/GOAL_FAILURE 真值交接:
        # 楼层预算耗尽时不得把多楼层任务困在本层——交接给真值引导楼梯路线, 可
        # 从当前走廊位姿安全重定位, 错过房间留给有界后期重访。严格
        # _stair_return_handoff_authorized()(terminal_return_latched)在此太刚性:
        # run ..._231 在 TIME_LIMIT 结束时 exited=2 且无走廊 latch, F1 静默退出,
        # 楼梯管理器永久挂在 STAIR_HANDOFF_NOT_REACHED。
        # 兜底时必须无条件释放 active_door, 不再要求严格闸 4/4: 3/4 时
        # active_door 保持活动会挡住松弛闸 _multi_floor_truth_handoff_safe 的
        # active_door 检查, 交接永不触发(批次17 RUN2 STAIR_HANDOFF_NOT_REACHED
        # 根因)。4/4 满足时行为不变。
        if (self.termination_reason in ("TIME_LIMIT", "GOAL_FAILURE") and
                self.stair_handoff_on_corridor_exit):
            if self.room_scheduler.active_door is not None:
                self.room_scheduler.abort_active_room(
                    self.elapsed(), "time_limit_truth_handoff_release")
                self.corridor_sweep_history.append({
                    "event": "TIME_LIMIT_ACTIVE_ROOM_ABORTED_FOR_STAIR_HANDOFF",
                    "elapsed_sec": round(self.elapsed(), 3),
                    "reason": self.termination_reason,
                })
        _truth_handoff_safe = getattr(
            self, "_multi_floor_truth_handoff_safe", None)
        if (self.termination_reason in ("TIME_LIMIT", "GOAL_FAILURE") and
                self.stair_handoff_on_corridor_exit and
                _truth_handoff_safe is not None and
                _truth_handoff_safe()):
            rospy.logwarn(
                "TIME_LIMIT/GOAL_FAILURE truth stair handoff ENGAGED "
                "(reason=%s, handoff_on_corridor_exit=%s)",
                self.termination_reason, self.stair_handoff_on_corridor_exit)
            fallback_reason = ("TIME_LIMIT_AFTER_ROOM_EXIT"
                               if self.termination_reason == "TIME_LIMIT"
                               else "GOAL_FAILURE_AFTER_CORRIDOR_RETURN")
            self.termination_reason = \
                "STAIR_CORRIDOR_EXIT_HANDOFF_TRUTH_REQUESTED"
            self.multi_floor_handoff_pending = True
            self._set_state("STAIR_LOBBY_HANDOFF",
                            self.termination_reason)
            self.cancel_goal_pub.publish(String(
                data="stair_truth_return_fallback_goal_failure"))
            self.corridor_sweep_history.append({
                "event": "STAIR_CORRIDOR_EXIT_HANDOFF_TRUTH_REQUESTED",
                "elapsed_sec": round(self.elapsed(), 3),
                "reason": fallback_reason,
                "policy": "truth_stair_handoff_preserves_next_floor",
            })
        elif self.termination_reason == "TIME_LIMIT":
            rospy.logwarn(
                "TIME_LIMIT epilogue without truth handoff "
                "(handoff_on_corridor_exit=%s)",
                self.stair_handoff_on_corridor_exit)
            self._set_state("TIME_LIMIT")
        if self.termination_reason == "TIME_LIMIT":
            self._set_state("TIME_LIMIT")
        self.exploration_end_elapsed = round(self.elapsed(), 3)
        self._set_state("SHUTDOWN", self.termination_reason or "SHUTDOWN")
        self._write_all_logs()
        self.finalize_result_pub.publish(Bool(data=True))
        self._generate_visualization()

    def _on_shutdown(self):
        if self.termination_reason is None:
            self.termination_reason = "SHUTDOWN"
        self._write_all_logs()
        self.finalize_result_pub.publish(Bool(data=True))
        # roslaunch invokes this callback when the user stops a run before
        # ``run()`` reaches its normal epilogue.  Generate the same offline
        # figures here so an interrupted/early-stopped run is still
        # inspectable.  The visualization script only needs the persisted
        # logs; map finalization is best-effort during ROS teardown.
        try:
            self._spawn_shutdown_visualization()
        except Exception as error:  # visualization must not block shutdown
            rospy.logerr("Detached visualization scheduling failed: %s", error)


if __name__ == "__main__":
    rospy.init_node("baseline_exploration_manager")
    try:
        BaselineExplorationManager().run()
    except Exception as error:  # explicit terminal evidence before ROS exits
        rospy.logfatal("Baseline exploration manager failed: %s", error)
        raise
