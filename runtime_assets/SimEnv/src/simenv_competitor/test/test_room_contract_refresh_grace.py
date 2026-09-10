#!/usr/bin/env python3
"""Regression checks for the bounded physical-partner refresh grace."""

import ast
import math
import pathlib


CORE = (pathlib.Path(__file__).resolve().parents[1] /
        "scripts" / "lightweight_room_core.py")
MANAGER = (pathlib.Path(__file__).resolve().parents[1] /
           "scripts" / "baseline_exploration_manager.py")
LAUNCH = (pathlib.Path(__file__).resolve().parents[1] /
          "launch" / "baseline_fastlio_exploration.launch")
UPPER_MANAGER = (pathlib.Path(__file__).resolve().parents[1] /
                 "scripts" / "second_floor_exploration_manager.py")


def _source():
    return CORE.read_text(encoding="utf-8")


def test_rgbd_hazard_projection_uses_truth_odometry_on_all_floors():
    launch = LAUNCH.read_text(encoding="utf-8")
    detector = launch[launch.index('type="red_ball_detector.py"'):
                      launch.index('</node>',
                                   launch.index('type="red_ball_detector.py"'))]
    assert '<param name="odom_topic" value="/simenv/second_floor_truth_odometry"/>' in detector
    assert '$(arg navigation_odom_topic)' not in detector


def test_refresh_grace_is_bounded_and_contract_only():
    source = _source()
    assert "mandatory_two_pose_refresh_grace_seconds: float = 18.0" in source
    assert "single_physical_view_owned = bool(" in source
    assert "locked_physical_contract and" in source
    assert "len(self.successful_roles.intersection({\"G3\", \"G4\"})) == 1" in source
    assert "not exhausted_physical_roles" in source


def test_f2_registration_loss_gets_bounded_truth_replan_before_abort():
    source = MANAGER.read_text(encoding="utf-8")
    assert "upper_floor_truth_retry = bool(" in source
    assert "self.floor_number >= 2 and" in source
    assert "self.allow_truth_exploration_when_registration_lost" in source
    assert "UPPER_FLOOR_TRUTH_DEGRADED_GOAL_REPLAN" in source
    assert "self.f3_truth_degraded_retry_count <" in source


def test_single_completed_view_gets_one_bounded_partner_window():
    source = _source()
    assert (
        "mandatory_two_pose_execution_retry_grace_seconds: float = 10.0"
        in source)
    assert 'execution_retry_armed = bool(' in source
    assert 'single_physical_view_owned = bool(' in source
    assert '"ROOM_TWO_POSE_SINGLE_VIEW_GRACE_ARMED"' in source
    assert '"one_physical_view_then_partner_or_partial"' in source


def test_locked_second_view_gets_bounded_heading_and_settle_margin():
    source = _source()
    marker = "A failed first G4 can leave the body laterally displaced"
    start = source.index(marker)
    end = source.index("if proposal is not None:", start)
    window = source[start:end]
    assert 'str(proposal.get("role", "")) == "G4"' in window
    assert '"open_deep_near"' in window
    assert '"obstacle_front_opposite_sides"' in window
    assert "min(float(self.config.room_goal_timeout_seconds)," in window
    assert "18.0" in window


def test_truth_portal_probe_cannot_starve_opposite_side():
    source = MANAGER.read_text(encoding="utf-8")
    start = source.index("def _plan_f3_truth_room_fallback")
    end = source.index("def _canonical_truth_portal_axis", start)
    planner = source[start:end]
    assert "deferred_probe = None" in planner
    assert "if deferred_probe is None:" in planner
    assert "deferred_probe = probe" in planner
    assert "return deferred_probe" in planner
    assert "return probe" not in planner


def test_refresh_remains_non_physical_evidence():
    source = _source()
    assert '"two_pose_strategy":\n                        "mandatory_two_pose_unavailable_map_refresh"' in source
    assert 'map_refresh_only = bool((goal.get("room_goal_diagnostic") or {}).get(' in source
    assert "not map_refresh_only" in source


def test_hard_deadline_still_runs_after_grace():
    source = _source()
    grace = source.index('"ROOM_TWO_POSE_SINGLE_VIEW_GRACE_ARMED"')
    deadline = source.index(
        "if elapsed >= observation_deadline and missing_physical_view:", grace)
    assert grace < deadline


def test_partial_open_room_preserves_single_measured_depth_class():
    source = _source()
    start = source.index("def _canonicalize_open_partial_contract")
    end = source.index("def _confirm_entry", start)
    canonicalizer = source[start:end]
    assert "available_physical_pose_depth" in canonicalizer
    assert "door.depth(deepest_pose) >= minimum_deep" in canonicalizer
    assert 'preserve_role = "G3"' in canonicalizer
    assert 'repair_role = "G4"' in canonicalizer


def test_refresh_cannot_fall_through_to_generic_adaptive_stop():
    source = _source()
    guard = source.index(
        '"ROOM_TWO_POSE_FINAL_FRONT_GAP_GUARD_SELECTED"')
    stopped = source.index(
        '"ADAPTIVE_ROOM_OBSERVATION_STOPPED"', guard)
    assert guard < stopped
    gate = source[source.rfind("if (proposal is None", 0, guard):guard]
    assert "mandatory_two_pose_pending" in gate
    assert "obstacle_front_opposite_sides" in gate
    assert "select_front_gap_opposite_side_second_visual_view" in gate


def test_first_obstacle_partner_proposal_is_geometry_gated():
    """The first G4 must be checked before any failed-attempt history exists."""
    source = _source()
    marker = "Validate the very first proposed second physical view as well"
    start = source.index(marker)
    end = source.index("# A physical two-view contract must not fall through", start)
    gate = source[start:end]
    assert 'str(proposal.get("role")) in ("G3", "G4")' in gate
    assert "self.two_pose_opposite_repair_required" in gate
    assert "proposal_behind_obstacle" in gate
    assert "proposal is not None and attempted and" not in gate


def test_direct_activation_rejects_completed_physical_door():
    source = _source()
    start = source.index("def _activate(self, door:")
    end = source.index("def request_visual_deepening", start)
    activation = source[start:end]
    assert "DIRECT_ACTIVATION_COMPLETED_DOOR_REJECTED" in activation
    assert "if existing.completed:" in activation
    assert "return None" in activation
    assert activation.index("if existing.completed:") < activation.index(
        'door.door_id = "estimated_door_')


def test_all_direct_activation_callers_handle_duplicate_rejection():
    source = _source()
    assert "TRUTH_PORTAL_STAGING_DUPLICATE_REJECTED" in source
    assert source.count("if activated is None:") >= 3


def test_obstacle_partner_window_uses_physical_lateral_speed():
    source = _source()
    # The contract flag is deliberately defined before the two-pose branch so
    # adaptive G4 proposals cannot hit an unbound local.  Include that setup in
    # the policy slice instead of assuming the literal contract name remains
    # below the explanatory comment.
    marker = "obstacle_contract = bool("
    start = source.index(marker)
    end = source.index("ROOM_TWO_POSE_MANDATORY_WINDOW_EXTENDED", start)
    budget = source[start:end]
    assert '"obstacle_front_opposite_sides"' in budget
    assert "1.0 / 0.30 if obstacle_contract" in budget
    assert "35.0, proposal_execution_window + 8.0" in budget


def test_deep_obstacle_contract_has_bounded_time_to_reach_both_peeks():
    source = _source()
    assert "deep_obstacle_contract_grace_seconds: float = 40.0" in source
    assert 'obstacle_view_policy(self.active_door) ==' in source
    assert '"deep_outer_side_peek"' in source
    assert "observation_deadline += max(" in source


def test_obstacle_execution_window_is_not_reclipped_to_generic_18_seconds():
    source = _source()
    marker = "The mandatory-window calculation above used the real"
    start = source.index(marker)
    end = source.index("# Preserve the bounded retry counter", start)
    goal = source[start:end]
    assert "if obstacle_contract else" in goal
    assert "max(self.config.room_goal_timeout_seconds, 35.0)" in goal


def test_deep_obstacle_exit_skips_second_dense_replay():
    source = _source()
    marker = "deep_obstacle_exit = bool("
    start = source.index(marker)
    end = source.index("return {\"success\": bool(effective_success)", start)
    recovery = source[start:end]
    assert '"deep_outer_side_peek"' in recovery
    assert "not effective_success" in recovery
    assert "self.active_door is not None" in recovery
    assert "effective_exit_retry_limit" in recovery
    assert "1 if deep_obstacle_exit" in recovery


def test_manager_sparsifies_verified_exit_trajectory_before_dispatch():
    manager = MANAGER.read_text(encoding="utf-8")
    marker = "dense_exit_anchor_count = len(anchors)"
    start = manager.index(marker)
    end = manager.index("if (anchors and not anchor_skipped", start)
    sparse = manager[start:end]
    assert "sparsify_verified_trace(" in sparse
    assert "1.35 if deep_obstacle_exit" in sparse
    assert "ROOM_EXIT_TRAJECTORY_SPARSIFIED" in sparse


def test_truth_portal_dispatch_failure_gets_one_fresh_map_retry():
    manager = MANAGER.read_text(encoding="utf-8")
    start = manager.index("def _rearm_failed_truth_portal_execution")
    end = manager.index("def _inferred_corridor_centerline_from_door", start)
    helper = manager[start:end]
    assert "truth_portal_execution_rearms" in helper
    assert '"mandatory_portal_waypoint_lost"' in helper
    assert '"scan_lite_refinement_failed"' in helper
    assert "retry_limit - 1" in helper
    assert "TRUTH_PORTAL_EXECUTION_SINGLE_RETRY_REARMED" in helper
    assert "self._rearm_failed_truth_portal_execution(" in manager


def test_truth_locked_f3_portal_keeps_revalidated_mandatory_anchors():
    manager = MANAGER.read_text(encoding="utf-8")
    start = manager.index("truth_locked_fallback_entry = bool(")
    end = manager.index("truth_obstacle_semantic_stale_map_fallback", start)
    guarded = manager[start:end]
    assert 'goal.get("truth_portal_fallback_key")' in guarded
    assert "2.75 if goal.get" in guarded
    assert "self._truth_layout_portal_segment_clear(" in guarded


def test_recent_exit_rearms_only_same_row_opposite_truth_portal():
    manager = MANAGER.read_text(encoding="utf-8")
    start = manager.index("def _rearm_recent_exit_opposite_truth_portal")
    end = manager.index("def _finish_bounded_missing_room_retrace", start)
    helper = manager[start:end]
    assert "opposite_side = -int(self.last_room_exit_side)" in helper
    assert "RECENT_EXIT_OPPOSITE_TRUTH_PORTAL_FRESHENED" in helper
    assert 'truth_room_fallback_last_grid_update", {}).pop(' in helper
    assert "physical_exit_new_reverse_view_preserves_attempt_budget" in helper
    assert "attempts[key] = retry_limit - 1" in helper
    assert "RECENT_EXIT_OPPOSITE_TRUTH_PORTAL_REARMED" in helper
    assert "self._rearm_recent_exit_opposite_truth_portal(exit_pose)" in manager
    assert '"~corridor_terminal_past_far_station_m", 1.2' in manager
    config = (MANAGER.parents[1] / "config" /
              "fuel_semantic_fastlio_exploration.yaml").read_text(
        encoding="utf-8"
    )
    assert "corridor_terminal_past_far_station_m: 1.2" in config


def test_fresh_exit_partner_stages_before_stationary_portal_prescan():
    """The physical partner gets first refusal without weakening ENTRY."""
    manager = MANAGER.read_text(encoding="utf-8")
    start = manager.index("fresh_opposite_partner = bool(")
    prescan = manager.index(
        "# A corridor-side probe changes the view through the aperture.",
        start)
    block = manager[start:prescan]
    assert "activate_truth_portal_staging(" in block
    assert "FRESH_EXIT_OPPOSITE_TRUTH_STAGING_FAST_PATH" in block
    assert '"stationary_prescan_skipped": True' in block
    assert '"physical_entry_still_required": True' in block
    assert '"live_scan_lite_3d_still_required": True' in block
    assert 'int(side) == -int(recent_exit_side)' in block
    assert manager.index("fresh_opposite_partner = bool(") < manager.index(
        "prescan_key =", start)


def test_completed_pair_chord_requires_one_fresh_unchanged_3d_audit():
    """A stale long corridor route may not be compacted from truth alone."""
    manager = MANAGER.read_text(encoding="utf-8")
    start = manager.index("completed_pair_stale_corridor = bool(")
    end = manager.index("upper_floor_truth_corridor_fallback = bool(", start)
    block = manager[start:end]
    assert "completed_pair_truth_chord_refresh_consumed" in block
    assert "completed_pair_truth_corridor_fresh_3d_audit" in block
    assert "repair_colliding_waypoints=False" in block
    assert "repair_start_waypoint=False" in block
    assert "direct_unchanged = bool(" in block
    assert "COMPLETED_NEAR_PAIR_FRESH_3D_CHORD_AUTHORIZED" in block
    assert 'path_result["execution_waypoints"] = list(original)' in block
    assert "fall_back_to_stable_segmented_route" in block
    assert '"waypoint_collision_replan_required"' in block
    assert "stable_route = execution_waypoints(" in block
    assert "COMPLETED_NEAR_PAIR_FRESH_3D_REJECTED_SEGMENTED" in block
    assert '"execute_bounded_truth_verified_segments"' in block
    assert block.count("return path_result") >= 2


def test_f1_completed_pair_carries_narrow_generated_corridor_authorization():
    """F1 may use this audited inter-row transit without global truth mode."""
    manager = MANAGER.read_text(encoding="utf-8")
    assert '"first_floor_generated_corridor_transit_authorized": bool(' in manager
    start = manager.index("completed_pair_stale_corridor = bool(")
    end = manager.index("if (completed_pair_stale_corridor", start)
    block = manager[start:end]
    assert '"first_floor_generated_corridor_transit_authorized"' in block
    assert "bool(self.offline_truth_layout_metadata)" in block
    assert "self.allow_truth_exploration_when_registration_lost or" in block


def test_generated_exit_elides_center_stop_only_after_matching_3d_audit():
    """Door centre remains an audited crossing constraint, not a brake."""
    manager = MANAGER.read_text(encoding="utf-8")
    start = manager.index("generated_matches_live_audit = bool(")
    end = manager.index("waypoints = execution_canonical", start)
    block = manager[start:end]
    assert 'goal.get("direct_portal_exit_chord_3d_verified")' in block
    assert "mandatory_exit[0]" in block
    assert "mandatory_exit[-1]" in block
    assert "execution_canonical = [canonical[0], canonical[-1]]" in block
    assert "ROOM_EXIT_GENERATED_TRUTH_CENTER_STOP_ELIDED" in block
    assert '"narrow_door_alignment_still_required": True' in block
    assert '"physical_door_plane_crossing_still_required":' in block


def test_verified_exit_rebased_centerline_releases_debt_before_truth_fallback():
    manager = MANAGER.read_text(encoding="utf-8")
    start = manager.index("def _plan_corridor_door_goal")
    end = manager.index("def _plan_f3_truth_room_fallback", start)
    planner = manager[start:end]
    assert "verified_exit_rebased_corridor_centerline" in planner
    assert "POST_EXIT_CENTERLINE_DEBT_RELEASED_AT_ROW" in planner
    assert planner.index("verified_exit_rebased_corridor_centerline") < \
        planner.index("measured_corridor_band_longitudinal_transit")
    assert planner.index("verified_exit_rebased_corridor_centerline") < \
        planner.index("_plan_f3_truth_room_fallback")


def test_validated_deferred_entry_releases_completed_exit_debt_on_centerline():
    """run95's fourth portal must not wait for another corridor sweep."""
    manager = MANAGER.read_text(encoding="utf-8")
    start = manager.index("def _plan_corridor_door_goal")
    end = manager.index("def _plan_f3_truth_room_fallback", start)
    planner = manager[start:end]
    assert "validated_deferred_centerline = bool(" in planner
    assert '"room_transaction_complete", False' in planner
    assert "deferred_prevalidated_door_after_centerline" in planner
    assert "deferred_prevalidated_entry_after_centerline" in planner
    assert "validated_deferred_exit_centerline" in planner
    assert planner.index("validated_deferred_centerline = bool(") < \
        planner.index("longitudinal_transit >= 0.40")


def test_recent_exit_truth_fallback_prioritizes_physical_partner_side():
    manager = MANAGER.read_text(encoding="utf-8")
    start = manager.index("def _plan_f3_truth_room_fallback")
    end = manager.index("def _canonical_truth_portal_axis", start)
    fallback = manager[start:end]
    assert "recent_exit_side = self._recent_room_exit_opposite_side(pose)" in fallback
    assert "preferred_side = -int(recent_exit_side)" in fallback
    assert "side_order.sort(key=lambda item: item[0] != preferred_side)" in fallback


def test_new_room_waits_for_previous_centerline_recovery():
    source = _source()
    start = source.index("def _activate(self, door:")
    end = source.index("DIRECT_ACTIVATION_COMPLETED_DOOR_REJECTED", start)
    guard = source[start:end]
    assert "NEW_DOOR_ACTIVATION_BLOCKED_PENDING_CENTERLINE" in guard
    assert "exit_then_centerline_then_next_physical_room" in guard


def test_prevalidated_opposite_door_resumes_after_centerline_recovery():
    """A briefly visible paired door must survive the mandatory recenter."""
    source = _source()
    promotion = source.index("LOCAL_DOOR_DEFERRED_UNTIL_CENTERLINE")
    confirmation = source.index(
        "def confirm_corridor_centerline_recovered", promotion)
    reactivation = source.index(
        "LOCAL_DOOR_REACTIVATED_AFTER_CENTERLINE", confirmation)
    assert promotion < confirmation < reactivation
    body = source[confirmation:reactivation]
    assert "self.pending_corridor_recovery_door = None" in body
    assert "deferred_prevalidated_door_after_centerline" in body
    assert "self._activate(" in body


def test_real_corridor_transit_releases_stale_centerline_debt():
    manager = MANAGER.read_text(encoding="utf-8")
    start = manager.index("def _plan_corridor_door_goal")
    end = manager.index("def _plan_f3_truth_room_fallback", start)
    planner = manager[start:end]
    assert "measured_corridor_band_longitudinal_transit" in planner
    assert "longitudinal_transit >= 0.40" in planner
    assert "pending_recovery = None" in planner
    assert "self._on_station_corridor_centerline(" in planner
    assert planner.index(
        "measured_corridor_band_longitudinal_transit") < planner.index(
        "_plan_f3_truth_room_fallback")


def test_deferred_measured_portal_preempts_expensive_truth_fallback():
    """A measured fourth door must finish EXIT recentering before truth A*."""
    manager = MANAGER.read_text(encoding="utf-8")
    start = manager.index("def _plan_corridor_door_goal")
    end = manager.index("def _plan_f3_truth_room_fallback", start)
    planner = manager[start:end]
    assert "deferred_prevalidated_door_after_centerline" in planner
    assert "DEFERRED_LOCAL_PORTAL_PREEMPTS_TRUTH_FALLBACK" in planner
    assert "if pending_recovery is None and" in planner
    assert "deferred_local_door is None else None" in planner
    assert planner.index("deferred_local_door = getattr(") < planner.index(
        "_plan_f3_truth_room_fallback")
    assert "DEFERRED_LOCAL_PORTAL_CENTERLINE_RECOVERY_SELECTED" in planner
    assert "same_station_short_recenter_before_owned_entry" in planner
    recenter = planner.index(
        "DEFERRED_LOCAL_PORTAL_CENTERLINE_RECOVERY_SELECTED")
    truth_fallback = planner.index("if truth_fallback is not None")
    assert recenter < truth_fallback


def test_reactivated_deferred_portal_dispatches_entry_in_same_cycle():
    """A promoted opposite door cannot be overwritten by a corridor probe."""
    manager = MANAGER.read_text(encoding="utf-8")
    helper_start = manager.index(
        "def _plan_reactivated_deferred_room_goal")
    planner_start = manager.index("def _plan_corridor_door_goal", helper_start)
    helper = manager[helper_start:planner_start]
    assert "_next_room_goal_with_contract(" in helper
    assert 'goal["scheduler_phase"] = "ROOM_ENTRY"' in helper
    assert '"DEFERRED_LOCAL_ROOM_ENTRY_DISPATCHED"' in helper
    assert "same_cycle_centerline_recovery_then_entry" in helper

    planner_end = manager.index("def _plan_f3_truth_room_fallback", planner_start)
    planner = manager[planner_start:planner_end]
    # Normal recenter, longitudinal-transit recovery, and the verified
    # post-EXIT/rebased-centreline fast path must all dispatch an already
    # promoted doorway in the same scheduling cycle.
    assert planner.count("return self._plan_reactivated_deferred_room_goal(") == 3
    assert "deferred_local_after_transit_centerline" in planner
    assert "deferred_local_after_lateral_centerline" in planner
    transit_confirmation = planner.index(
        "measured_corridor_band_longitudinal_transit")
    transit_dispatch = planner.index(
        "deferred_local_after_transit_centerline", transit_confirmation)
    truth_fallback = planner.index("_plan_f3_truth_room_fallback")
    assert transit_confirmation < transit_dispatch < truth_fallback


def test_direct_entry_aligns_to_portal_station_before_wall_approach():
    """A longitudinally offset robot must not cut diagonally into the jamb."""
    source = _source()
    marker = "station_aligned_anchor = ["
    start = source.index(marker)
    end = source.index("direct_anchors =", start)
    alignment = source[start:end]
    # Portal-centre repair can retain the measured normal plane while repairing
    # a door-edge tangent error.  Alignment therefore has to be measured in the
    # selected physical portal frame, not the stale EstimatedDoorway frame.
    lateral_start = source.rindex("current_lateral = (", 0, start)
    lateral_geometry = source[lateral_start:start]
    assert "portal_center[0]) * tangent[0]" in lateral_geometry
    assert "portal_center[0] + current_depth * nx" in alignment
    assert "station_alignment_required = bool(" in alignment
    assert "abs(float(current_lateral)) > centreline_half_width" in alignment
    assert "approach_anchor = (station_aligned_anchor" in alignment


def test_deep_obstacle_first_view_survives_stale_2d_stripe():
    """F3 room2 must emit a real deep side target, never an in-place G3."""
    source = _source()
    start = source.index("def select_truth_obstacle_gap_first_visual_view")
    end = source.index("def select_mandatory_second_visual_view", start)
    selector = source[start:end]
    assert "ordered_side_targets = tuple(sorted(" in selector
    assert "value * current_lateral < 0.0" in selector
    assert '"preflight_path": ([] if known_occupied else' in selector
    assert '"stale_raster_segment_veto_overridden": bool(' in selector
    assert "if stale_raster_candidate is not None:" in selector


def test_truth_staged_door_survives_centerline_and_blocks_partial_handoff():
    """A known fourth portal cannot be discarded by the 3/4 escape gate."""
    core = _source()
    assert "TRUTH_PORTAL_STAGING_DEFERRED_UNTIL_CENTERLINE" in core
    assert "deferred_entry_waypoints_after_centerline" in core
    staging_start = core.index("def activate_truth_portal_staging")
    staging_end = core.index("def retry_known_unvisited_door", staging_start)
    staging = core[staging_start:staging_end]
    # An ordinary preflight rejection may have changed the scheduler to
    # CORRIDOR_SWEEP.  Deferring the validated opposite portal must restore
    # ownership to the previous EXIT's centreline transaction, otherwise the
    # deferred portal can never be promoted.
    assert '"CORRIDOR_RESUME", now' in staging
    assert "truth_portal_deferred_pending_centerline" in staging
    assert staging.index('"CORRIDOR_RESUME", now') < staging.index(
        '"TRUTH_PORTAL_STAGING_DEFERRED_UNTIL_CENTERLINE"')

    manager = MANAGER.read_text(encoding="utf-8")
    known_start = manager.index("def _known_unvisited_room_doors")
    known_end = manager.index(
        "def _terminal_return_has_blocking_known_door", known_start)
    known = manager[known_start:known_end]
    assert "deferred_prevalidated_door_after_centerline" in known
    assert "doors.append(deferred)" in known

    fallback_start = manager.index("def _multi_floor_truth_handoff_safe")
    fallback_end = manager.index("def _stair_return_deadline", fallback_start)
    fallback = manager[fallback_start:fallback_end]
    assert "pending_known_doors = self._known_unvisited_room_doors()" in fallback
    assert "if pending_known_doors:" in fallback
    # A pending fourth portal blocks an ordinary early handoff, but after an
    # explicitly armed bounded emergency it remains a strict defect rather
    # than stranding all later floors and the final descent.
    assert "if pending_known_doors and not emergency_continuation:" in fallback
    assert "if pending_known_doors:" in fallback
    assert fallback.index("emergency_partial_handoff_armed") < fallback.index(
        "if pending_known_doors and not emergency_continuation:")
    assert "preserving %d pending" in fallback


def test_locked_open_room_truth_route_is_available_on_first_floor():
    """F1 far-room stale raster must not discard a physically clear G3."""
    manager = MANAGER.read_text(encoding="utf-8")
    helper_start = manager.index(
        "def _truth_layout_open_room_segment_clear")
    helper_end = manager.index(
        "def _truth_layout_obstacle_gap_segment_clear", helper_start)
    helper = manager[helper_start:helper_end]
    assert "self.floor_number < 2" not in helper
    assert "not self.allow_truth_exploration_when_registration_lost" not in helper
    assert "viewpoint_contract_locked_before_entry" in helper
    assert "truth_furniture_footprint_blocked" in helper

    obstacle_start = helper_end
    obstacle_end = manager.index("def _plan_path", obstacle_start)
    obstacle_helper = manager[obstacle_start:obstacle_end]
    # The same strict helper is required on F1 as well: run74 showed that a
    # truth-matched, physically entered obstacle room can have a stale 2-D gap
    # stripe on the first floor too.  Authorization still requires a locked
    # contract, physical truth crossing and the ordinary dense live 3-D audit.
    assert "self.floor_number < 2" not in obstacle_helper
    assert "not self.allow_truth_exploration_when_registration_lost" not in obstacle_helper
    assert "active_door.entry_truth_crossing_confirmed" in obstacle_helper
    assert 'active_door.viewpoint_contract !=' in obstacle_helper
    assert '"obstacle_front_opposite_sides"' in obstacle_helper

    fallback_start = manager.index(
        "truth_open_semantic_stale_map_fallback = bool(")
    fallback_end = manager.index(
        "stale_two_pose_direct_disconnect", fallback_start)
    fallback = manager[fallback_start:fallback_end]
    assert "self.floor_number >= 2" not in fallback
    assert "OPEN_ROOM_STALE_MAP_COLLISION_TRUTH_ROUTE_OVERRIDE" in fallback


def test_first_floor_truth_portal_fallback_is_not_blocked_by_upper_floor_flag():
    manager = MANAGER.read_text(encoding="utf-8")
    start = manager.index("def _plan_f3_truth_room_fallback")
    end = manager.index("def _canonical_truth_portal_axis", start)
    fallback = manager[start:end]
    assert ("not self.upper_floor_truth_room_fallback_active and\n"
            "             not first_floor_truth_portal_authorized" in fallback)


def test_truth_validated_open_view_authorizes_executor_relative_anchor():
    manager_source = MANAGER.read_text(encoding="utf-8")
    execute_start = manager_source.index("def _execute_path")
    execute_end = manager_source.index("def _refine_path", execute_start)
    execute = manager_source[execute_start:execute_end]
    assert "OPEN_ROOM_EXECUTOR_TRUTH_ANCHOR_AUTHORIZED" in execute
    assert "OPEN_ROOM_EXECUTOR_TRUTH_ROUTE_AUTHORIZED" in execute
    assert "OBSTACLE_GAP_EXECUTOR_TRUTH_ROUTE_AUTHORIZED" in execute
    assert "_truth_layout_open_room_segment_clear" in execute
    assert 'room_role in ("G3", "G4")' in execute
    assert "not obstacle_room_route" in execute
    assert "waypoints = [final_point]" in execute
    assert "dense_truth_clear_direct_physical_chord_no_credit" in execute
    assert "truth_relative_navigation=bool(" in execute

    executor_source = (MANAGER.parent / "goal_executor.py").read_text(
        encoding="utf-8")
    assert '"/simenv/goal_truth_relative_navigation"' in executor_source
    timer_start = executor_source.index("def _on_timer")
    timer = executor_source[timer_start:]
    assert "semantic_truth_authorized = bool(" in timer
    assert "truth_navigation_authorized" in timer
    on_goal_start = executor_source.index("def _on_goal")
    on_goal_end = executor_source.index(
        "def _on_locomotion_ready", on_goal_start)
    on_goal = executor_source[on_goal_start:on_goal_end]
    assert "semantic_truth_anchor_refreshed_for_goal" in on_goal


def test_provisional_terminal_latch_cannot_rewind_an_incomplete_floor():
    manager_source = MANAGER.read_text(encoding="utf-8")
    start = manager_source.index("terminal_return_active = bool(")
    end = manager_source.index("returning_home = bool(", start)
    gate = manager_source[start:end]
    assert "all_rooms_exited or" in gate
    assert "emergency_partial_handoff_armed" in gate
    assert "corridor_partial_return_trigger_reason" in gate


def test_stationary_hold_limit_only_releases_f1_obligation():
    manager_source = MANAGER.read_text(encoding="utf-8")
    start = manager_source.index("def _recent_room_exit_opposite_side")
    end = manager_source.index("def _recent_exit_needs_opposite_check", start)
    obligation = manager_source[start:end]
    assert "self.floor_number < 2" in obligation
    assert '"post_exit_opposite_station_holds", 0' in obligation
    assert "return int(self.last_room_exit_side)" in obligation


def test_upper_floor_pair_uses_truth_portal_before_station_wait():
    manager_source = MANAGER.read_text(encoding="utf-8")
    event = manager_source.index(
        "POST_EXIT_TRUTH_OPPOSITE_SELECTED")
    start = manager_source.rfind(
        "if (goal is None and self.floor_number >= 2)", 0, event)
    hold = manager_source.index(
        "POST_EXIT_OPPOSITE_ROOM_STATION_HOLD", start)
    block = manager_source[start:hold]
    assert "self.floor_number >= 2" in block
    assert "self._plan_f3_truth_room_fallback(" in block
    assert "post_exit_opposite_station_hold_limit" in block


def test_prevalidated_pair_dispatches_same_cycle_before_station_hold():
    """run96: a prepared opposite ENTRY must not pay another visual hold."""
    manager_source = MANAGER.read_text(encoding="utf-8")
    event = manager_source.index(
        "POST_EXIT_DEFERRED_ENTRY_SAME_CYCLE_SELECTED")
    start = manager_source.rfind("deferred_ready_now = bool(", 0, event)
    hold = manager_source.index(
        "POST_EXIT_OPPOSITE_ROOM_STATION_HOLD", event)
    block = manager_source[start:hold]
    assert '"deferred_prevalidated_door_after_centerline"' in block
    assert '"deferred_prevalidated_entry_after_centerline"' in block
    assert "self._plan_corridor_door_goal()" in block
    assert "prevalidated_entry_consumes_exit_" in block


def test_both_obstacle_contracts_first_exit_skip_full_room_breadcrumb_replay():
    core = _source()
    # obstacle_view_policy is also a module-level selector called earlier in
    # next_goal(); assigning that name locally makes Python treat every call
    # in the method as an unbound local (run123).
    assert "        obstacle_view_policy = " not in core
    assert "obstacle_front_gap_direct_exit = bool(" in core
    direct_start = core.index("obstacle_front_gap_direct_exit = bool(")
    direct_end = core.index("fallback_exit = bool(", direct_start)
    direct = core[direct_start:direct_end]
    assert "self.exit_attempts == 0" in direct
    assert '"shallow_front_chord", "deep_outer_side_peek"' in direct
    assert "exit_portal is None" not in direct.split(
        "if obstacle_front_gap_direct_exit:", 1)[0]
    assert "mandatory_exit = [" in direct
    assert "historical room anchor" in direct
    assert "sent G4 back across the gap" in direct
    assert "truth_bounded_deep_obstacle_front_gap_exit" in core
    assert "ROOM_OBSTACLE_FRONT_GAP_DIRECT_EXIT_SELECTED" in core
    assert "ROOM_DEEP_OBSTACLE_SIDE_PRESERVING_EXIT_STAGED" in core
    assert "side_preserving_stage = self.active_door.contract_point(" in core
    assert 'goal["deep_obstacle_front_gap_segmented_exit"]' in core
    # run122 F3 room4 was a valid deep side-peek at depth 3.33 m in
    # front of a 3.80 m obstacle, but the shallow clearance margin rejected
    # the direct EXIT by about 8 cm and wasted 26 simulated seconds.
    assert ('0.15 if active_obstacle_view_policy ==' in core and
            '"deep_outer_side_peek" else' in core)
    assert "float(truth_front_depth) - obstacle_front_exit_margin" in direct
    manager = MANAGER.read_text(encoding="utf-8")
    start = manager.index("prepared_result = goal.get(")
    end = manager.index("verified_exit_backtrack = bool(", start)
    guard = manager[start:end]
    assert 'goal.get("deep_obstacle_front_gap_direct_exit")' in guard
    assert "truth_bounded_deep_obstacle_front_gap_exit" in guard
    assert 'goal.get("deep_obstacle_front_gap_segmented_exit")' in manager
    assert "point[1]) - float(current[1])" in manager


def test_truth_restage_exit_does_not_replay_stale_deep_anchor():
    core = _source()
    assert "truth_portal_restage_exit_pending = False" in core
    assert "ROOM_EXIT_TRUTH_RESTAGE_DIRECT_PORTAL_SELECTED" in core
    assert "truth_restage_direct_physical_portal_exit" in core
    assert "post_truth_reseat_inside_center_to_portal_to_corridor_" in core

    manager = MANAGER.read_text(encoding="utf-8")
    assert "scheduler.truth_portal_restage_exit_pending = True" in manager
    assert 'goal.get("truth_portal_restage_exit")' in manager
    assert "EXIT_TRUTH_RESTAGE_EXECUTOR_TRUTH_ANCHOR_AUTHORIZED" in manager
    assert "immutable_truth_portal_route_physical_motion_no_credit" in manager
    assert "ROOM_EXIT_TRUTH_RESTAGE_REFINER_BYPASSED" in manager
    assert "truth_restage_immutable_portal_route" in manager
    assert "ROOM_EXIT_TRUTH_RESTAGE_REACHED_PREFIX_SKIPPED" in manager
    assert "truth_virtual_map_pose" in manager
    assert "dispatch_point = (" in manager
    assert '"truth_restage_direct_physical_portal_exit"' in manager
    assert "not truth_portal_restage_exit and" in core
    assert "room_portal_crossing_minimum_speed" in manager


def test_deep_obstacle_direct_exit_uses_truth_relative_dispatch():
    """run121: a vetted deep-gap EXIT must not wait in stale map axes."""
    manager = MANAGER.read_text(encoding="utf-8")
    marker = manager.index(
        "DEEP_OBSTACLE_EXIT_EXECUTOR_TRUTH_ANCHOR_AUTHORIZED")
    start = manager.rfind("deep_truth_portal_exit = bool(", 0, marker)
    block = manager[start:marker]
    assert 'goal.get("deep_obstacle_front_gap_direct_exit")' in block
    assert '"truth_bounded_deep_obstacle_front_gap_exit"' in block
    assert "self._truth_contract_dispatch(" in block
    dispatch = manager[marker:manager.index(
        "elif generated_truth_portal_exit:", marker)]
    assert '"physical_exit_still_required": True' in dispatch
    assert "physical_crossing_no_credit" in dispatch


def test_generated_truth_entry_uses_one_immutable_physical_frame_contract():
    """run79: a safe generated portal must not execute in stale map/odom mix."""
    manager = MANAGER.read_text(encoding="utf-8")
    assert "def _live_truth_contract_pose(" in manager
    assert "def _truth_contract_dispatch(" in manager
    assert "def _truth_world_yaw_to_contract_map(" in manager
    assert "ROOM_ENTRY_GENERATED_TRUTH_PORTAL_REFINER_BYPASSED" in manager
    assert "ROOM_ENTRY_GENERATED_TRUTH_PORTAL_DYNAMIC_DISPATCH" in manager
    assert "ROOM_ENTRY_GENERATED_TRUTH_PORTAL_REACHED_PREFIX_SKIPPED" in manager
    assert "ROOM_ENTRY_GENERATED_TRUTH_CROSSING_AUDIT" in manager
    assert 'goal["generated_truth_entry_start_contract_pose"]' in manager
    assert 'semantic_final_point = truth_contract[:2]' in manager
    assert '"entry_truth_door_plane_not_crossed"' in manager
    assert '"physical_motion_and_truth_door_plane_required"' in manager

    start = manager.index(
        "generated_truth_portal_live_3d_clear = False")
    bypass = manager.index(
        "ROOM_ENTRY_GENERATED_TRUTH_PORTAL_REFINER_BYPASSED", start)
    gate = manager[start:bypass]
    assert "generated_truth_portal_live_3d_clear" in gate
    assert "self._scan_check(" in gate
    assert "math.ceil(length / 0.10)" in gate
    assert "if generated_truth_portal_live_3d_clear:" in gate


def test_far_row_truth_probe_uses_latest_physical_room_frame():
    """run80: a map-frame corridor probe must not preserve EXIT lateral drift."""
    manager = MANAGER.read_text(encoding="utf-8")
    assert "def _room_contract_transform_for_floor(" in manager
    assert 'bool(getattr(door, "visited", False))' in manager
    assert '"truth_coordinate_transform":\n                self._room_contract_transform_for_floor()' in manager
    assert "corridor_truth_probe = bool(" in manager
    assert "CORRIDOR_TRUTH_PROBE_DYNAMIC_DISPATCH" in manager
    assert "generated_corridor_contract_delta_in_live_odom_frame_" in manager


def test_truth_contract_dispatch_rotates_position_and_heading_consistently():
    """run81: translated-only portal deltas turn sideways after yaw drift."""
    manager = MANAGER.read_text(encoding="utf-8")
    start = manager.index("def _truth_contract_dispatch(")
    end = manager.index("def _room_contract_transform_for_floor(", start)
    dispatch = manager[start:end]
    assert "frame_rotation = math.atan2(" in dispatch
    assert "float(current_odom[3]) - truth_contract[3]" in dispatch
    assert "self._rotate_contract_delta_to_odom(" in dispatch
    assert '"contract_to_odom_rotation_rad"' in dispatch

    tree = ast.parse(manager)
    helper_node = next(
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and
        node.name == "_rotate_contract_delta_to_odom")
    helper_node.decorator_list = []
    namespace = {"math": math}
    exec(compile(ast.fix_missing_locations(ast.Module(
        body=[helper_node], type_ignores=[])), str(MANAGER), "exec"),
         namespace)
    rotate = namespace["_rotate_contract_delta_to_odom"]
    x_value, y_value = rotate(1.0, 0.0, math.pi / 2.0)
    assert abs(x_value) < 1e-9
    assert abs(y_value - 1.0) < 1e-9
    x_value, y_value = rotate(0.0, 2.0, -math.pi / 2.0)
    assert abs(x_value - 2.0) < 1e-9
    assert abs(y_value) < 1e-9


def test_obstacle_partial_retry_cannot_keep_two_invalid_roles():
    """run81: an invalid G4 must not make re-entry jump directly to EXIT."""
    source = _source()
    assert "def _canonicalize_obstacle_partial_contract(" in source
    start = source.index("def _canonicalize_obstacle_partial_contract(")
    end = source.index("def _confirm_entry(", start)
    canonicalizer = source[start:end]
    assert 'self.successful_roles.discard("G3")' in canonicalizer
    assert 'self.successful_roles.discard("G4")' in canonicalizer
    assert "maximum_gap_depth" in canonicalizer
    assert "outside_obstacle_side" in canonicalizer
    assert "ROOM_OBSTACLE_CONTRACT_PARTIAL_CANONICALIZED" in canonicalizer
    exit_start = source.index('if role == "EXIT":')
    exit_end = source.index("deep_obstacle_exit = bool(", exit_start)
    exit_block = source[exit_start:exit_end]
    assert "self._canonicalize_obstacle_partial_contract(" in exit_block


def test_first_obstacle_partner_gate_does_not_depend_on_repair_latch():
    """The locked contract itself is sufficient to reject a bad first G4."""
    source = _source()
    marker = "Validate the very first proposed second physical view as well"
    start = source.index(marker)
    end = source.index("# A physical two-view contract must not fall through", start)
    gate = source[start:end]
    assert '(self.two_pose_opposite_repair_required or' in gate
    assert '"obstacle_front_opposite_sides") and' in gate


def test_short_generated_exit_uses_audited_rigid_portal_dispatch():
    """run82: valid obstacle-side views must not EXIT in stale raster axes."""
    manager = MANAGER.read_text(encoding="utf-8")
    assert "generated_truth_portal_exit = False" in manager
    assert "ROOM_EXIT_GENERATED_TRUTH_PORTAL_CHAIN_SELECTED" in manager
    assert "ROOM_EXIT_GENERATED_TRUTH_PORTAL_DYNAMIC_DISPATCH" in manager
    assert 'goal["generated_truth_portal_exit"] = True' in manager
    marker = manager.index(
        "ROOM_EXIT_GENERATED_TRUTH_PORTAL_CHAIN_SELECTED")
    block = manager[manager.rfind(
        "generated_truth_portal_exit = False", 0, marker):marker]
    assert "self._truth_layout_portal_segment_clear(" in block
    assert "self._truth_layout_open_room_segment_clear(" in block
    assert "allow_locked_exit=True" in block
    assert "self._scan_check(" in block
    assert "shallow_front_chord" in block
    assert "deep_outer_side_peek" not in block
    dispatch = manager[marker:manager.index(
        "elif generated_truth_portal_entry:", marker)]
    assert "physical_exit_still_required" in dispatch
    assert "ROOM_EXIT_GENERATED_TRUTH_PORTAL_CHAIN_REJECTED" in manager


def test_short_generated_exit_has_dedicated_locked_exit_authorization():
    """run84: EXIT must not be rejected by helpers restricted to G3/G4."""
    manager = MANAGER.read_text(encoding="utf-8")
    helper_start = manager.index(
        "def _truth_layout_open_room_segment_clear(")
    helper_end = manager.index(
        "def _truth_layout_obstacle_gap_segment_clear(", helper_start)
    helper = manager[helper_start:helper_end]
    assert "allow_locked_exit=False" in helper
    assert 'allow_locked_exit and room_role == "EXIT"' in helper
    assert "active_door.entry_truth_crossing_confirmed" in helper
    assert "active_door.truth_room_id" in helper
    assert "self.room_scheduler.active_room_id is not None" in helper
    assert 'str(goal.get("estimated_door_id", ""))' in helper
    marker = manager.index(
        "ROOM_EXIT_GENERATED_TRUTH_PORTAL_CHAIN_SELECTED")
    block = manager[manager.rfind(
        "generated_truth_portal_exit = False", 0, marker):marker]
    assert "allow_locked_exit=True" in block
    assert "It is not another" in block


def test_verified_portal_disconnect_keeps_collision_free_entry_trace():
    """run48 F2 opposite ENTRY was safe at every anchor, but disconnected."""
    manager = MANAGER.read_text(encoding="utf-8")
    start = manager.index("verified_truth_portal_disconnect = bool(")
    end = manager.index("upper_floor_truth_corridor_fallback = bool(", start)
    block = manager[start:end]
    assert 'refined.reason == "disconnected_refined_segment"' in block
    assert "not refined.colliding_original_waypoints" in block
    assert 'goal.get("portal_preflight_verified", False)' in block
    assert "segmented_execution_waypoints(" in block
    assert "truth_verified_portal_disconnect_fallback" in block


def test_truth_matched_entry_can_override_only_one_contradictory_stale_voxel():
    """run85: a stale jamb voxel must not hide both generated far rooms."""
    manager = MANAGER.read_text(encoding="utf-8")
    marker = "generated_truth_portal_stale_live_3d_override = False"
    start = manager.index(marker)
    end = manager.index("# A mirrored paired-room ENTRY", start)
    block = manager[start:end]
    assert "not generated_truth_portal_live_3d_clear" in block
    assert 'live_3d_rejection.get("map_available")' in block
    assert 'live_3d_rejection.get("occupied_collision")' in block
    assert 'getattr(active_door, "truth_room_id", None)' in block
    assert "len(generated_truth_portal_prevalidation_details) == 3" in block
    assert "ROOM_ENTRY_GENERATED_TRUTH_PORTAL_STALE_3D_OVERRIDE" in block
    assert "physical_entry_still_required" in block


def test_f1_incomplete_corridor_can_use_bounded_truth_motion_to_far_row():
    """run48 must not call a stale raster wall the corridor end at 2/4."""
    manager = MANAGER.read_text(encoding="utf-8")
    probe_start = manager.index("def _plan_upper_floor_truth_corridor_probe")
    probe_end = manager.index("def _plan_f3_initial_truth_corridor_seed",
                              probe_start)
    probe = manager[probe_start:probe_end]
    assert "first_floor_truth_motion_authorized = bool(" in probe
    assert "self.offline_truth_layout_metadata" in probe
    assert "self.corridor_established" in probe
    assert "offline_corridor_allows_probe" in probe
    assert "floor_num < 1" in probe
    assert "corridor_truth_probe" in probe
    assert "self._room_contract_transform_for_floor()" in probe
    assert "self._contract_map_point_to_truth_world(" in probe
    assert "start_world" in probe
    assert "target_world" in probe

    schedule_start = manager.index("elif corridor_forward_lock:")
    schedule_end = manager.index("else:\n                    goal = self._plan_room_goal()",
                                 schedule_start)
    schedule = manager[schedule_start:schedule_end]
    assert "self.floor_number == 1" in schedule
    assert "self._plan_upper_floor_truth_corridor_probe(" in schedule


def test_f1_truth_partner_remains_physically_gated_before_station_wait():
    manager = MANAGER.read_text(encoding="utf-8")
    event = manager.index("POST_EXIT_TRUTH_OPPOSITE_SELECTED")
    hold = manager.index("POST_EXIT_OPPOSITE_ROOM_STATION_HOLD", event)
    block = manager[event:hold]
    assert "self.floor_number == 1" in block
    assert "self.offline_truth_layout_metadata" in block
    assert "self.corridor_established" in block
    assert "self._plan_f3_truth_room_fallback(" in block
    assert "live_physical_" in block
    assert "entry_gates_before_station_wait" in block


def test_partial_far_end_abort_waits_for_deferred_opposite_door_transaction():
    """run85: a fourth door pending centreline recovery must execute first."""
    manager = MANAGER.read_text(encoding="utf-8")
    start = manager.index("deferred_opposite_transaction_pending = bool(")
    end = manager.index("self._set_state(\n                    \"CORRIDOR_NO_FRONTIER_BOUNDED_ABORT\"",
                        start)
    block = manager[start:end]
    assert '"pending_corridor_recovery_door"' in block
    assert '"deferred_prevalidated_door_after_centerline"' in block
    assert '"deferred_prevalidated_entry_after_centerline"' in block
    assert "not deferred_opposite_transaction_pending" in block
    assert "UPPER_FLOOR_PARTIAL_FAR_END_ABORT" in block


def test_truth_matched_local_portal_reaches_live_3d_audit_after_2d_rejection():
    """run86: an exact local/truth door must not die before 3-D auditing."""
    manager = MANAGER.read_text(encoding="utf-8")
    marker = "ROOM_ENTRY_TRUTH_MATCHED_LOCAL_PORTAL_2D_"
    event = manager.index(marker)
    start = manager.rindex(
        "if (not result.get(\"success\") and", 0, event)
    end = manager.index(
        "# The 2-D circular inflation is intentionally conservative", event)
    block = manager[start:end]
    assert 'room_diagnostic.get(\n                            "truth_portal_center_aligned")' in block
    assert 'getattr(active_door, "truth_room_id", None)' in block
    assert "len(mandatory) == 3" in block
    assert "self._truth_layout_portal_segment_clear(" in block
    assert "len(truth_details) == 3" in block
    assert 'goal["truth_layout_portal_override"] = True' in block
    assert "physical_entry_still_required" in block


def test_truth_corridor_probe_obeys_terminal_retrace_and_entry_bounds():
    """run86: truth probes cannot bypass the bounded reverse window."""
    manager = MANAGER.read_text(encoding="utf-8")
    start = manager.index("def _plan_upper_floor_truth_corridor_probe")
    end = manager.index("def _plan_f3_initial_truth_corridor_seed", start)
    block = manager[start:end]
    assert "self._finish_bounded_missing_room_retrace(pose)" in block
    assert "self.terminal_missing_room_retrace_start_station" in block
    assert "self.terminal_missing_room_retrace_max_distance" in block
    assert "self.corridor_entry_anchor is not None" in block
    assert "remaining_to_entry" in block
    assert "advance = min(float(advance), remaining)" in block
    assert "TRUTH_CORRIDOR_PROBE_RETRACE_CLAMPED" in block
    assert "TRUTH_CORRIDOR_PROBE_RETRACE_NO_SAFE_ADVANCE" in block


def test_fresh_opposite_deferred_portal_skips_redundant_pi_prescan():
    """run120: deferred means centreline debt, not portal rejection."""
    manager = MANAGER.read_text(encoding="utf-8")
    start = manager.index("deferred_matches_candidate = bool(")
    end = manager.index(
        "# A corridor-side probe changes the view through the aperture.",
        start)
    block = manager[start:end]
    # The attribute is loaded immediately before the focused block and the
    # block must gate promotion on the resulting bounded recovery object.
    assert "pending_recovery is not None" in block
    assert "room_transaction_complete" in block
    assert "self._on_station_corridor_centerline(" in block
    assert "confirm_corridor_centerline_recovered(" in block
    assert "self._plan_reactivated_deferred_room_goal(" in block
    assert "FRESH_EXIT_OPPOSITE_DEFERRED_FAST_PATH" in block
    assert "FRESH_EXIT_OPPOSITE_DEFERRED_CENTERLINE_PENDING" in block
    assert '"stationary_prescan_skipped": True' in block
    assert "physical_entry_still_required" in block
    assert "live_scan_lite_3d_still_required" in block


def test_truth_probe_success_uses_actual_dispatched_endpoint_distance():
    """run120: repaired 0.75 m probe advanced 0.486 m and was not fake."""
    manager = MANAGER.read_text(encoding="utf-8")
    start = manager.index("executor_goal = (result.get(\"goal\")")
    end = manager.index("if (reported_success and", start)
    block = manager[start:end]
    assert "dispatched_probe_distance" in block
    assert "effective_probe_request = min(" in block
    assert "0.55 * effective_probe_request" in block
    assert "self.reached_tolerance + 0.08" in block
    assert '"effective_requested_advance_m"' in block


def test_cached_room_contract_uses_audit_only_refiner_before_fallback():
    """Cached G3/G4/EXIT geometry must not be changed by offset search."""
    manager = MANAGER.read_text(encoding="utf-8")
    start = manager.index("scheduler_preflight_room_contract = bool(")
    end = manager.index("elif (truth_disconnected_open_path or", start)
    block = manager[start:end]
    assert '"lightweight_room_semantic"' in block
    assert '"lightweight_room_exit"' in block
    assert "scheduler_preflight_room_contract" in block
    assert "repair_colliding_waypoints=False" in block
    assert "repair_start_waypoint=False" in block
    assert "no collision threshold relaxation" in block


def test_upper_floor_truth_probe_fails_fast_to_existing_truth_fallback():
    manager = MANAGER.read_text(encoding="utf-8")
    start = manager.index(
        'elif source == "upper_floor_truth_corridor_probe":')
    end = manager.index("elif scheduler_preflight_room_contract:", start)
    block = manager[start:end]
    assert "repair_colliding_waypoints=False" in block
    assert "repair_start_waypoint=False" in block
    # Both F1 and upper-floor verified-route fallbacks must recognize the
    # audit-only collision result produced by this branch.
    assert manager.count('"waypoint_collision_replan_required") and') >= 2


def test_truth_probe_dense_audit_samples_are_not_execution_stops():
    """run163's 16 audit samples must compact to one physical command."""
    manager = MANAGER.read_text(encoding="utf-8")
    start = manager.index("truth_corridor_probe_chord_compacted = bool(")
    end = manager.index("continuous_door_center_transit = False", start)
    block = manager[start:end]
    assert "refined.success" in block
    assert "not refined.repaired_waypoints" in block
    assert 'source == "upper_floor_truth_corridor_probe"' in block
    assert 'goal.get("corridor_truth_probe", False)' in block
    assert "len(original) == 1" in block
    assert "refined_execution = [tuple(original[-1])]" in block
    assert "UPPER_FLOOR_TRUTH_PROBE_AUDITED_CHORD_COMPACTED" in block
    assert "collision_checks" in block


def test_truth_probe_can_clear_only_a_verified_live_self_footprint():
    """run164 self voxel may not reject a continuous corridor pose."""
    manager = MANAGER.read_text(encoding="utf-8")
    start = manager.index("verified_portal_start = bool(")
    end = manager.index("room_truth_live_start = bool(", start)
    block = manager[start:end]
    assert '"upper_floor_truth_corridor_probe"' in block
    assert "corridor_live_start_override_allowed(" in block
    assert "registration_healthy" in block
    assert "start_override_radius = min(" in block
    assert "self.scan_config.body_collision_radius" in block
    assert "return_degraded" not in block


def test_truth_corridor_probe_uses_transit_speed_not_search_default():
    """A verified 13 m row transit must not inherit the 0.65 m/s search cap."""
    manager = MANAGER.read_text(encoding="utf-8")
    start = manager.index("def _publish_speed_for_goal")
    end = manager.index("def _publish_waypoint", start)
    block = manager[start:end]
    transit = block[block.index(
        'elif source in ("corridor_resume_centerline"'):
        block.index('elif source == "corridor_sweep"')]
    assert '"upper_floor_truth_corridor_probe"' in transit
    # Keep the F1 straight-line stability cap and GoalExecutor's corridor
    # degraded-localization cap attached to this newly classified source.
    f1_cap = block[block.index(
        'if (int(getattr(self, "floor_number", 1)) == 1'):]
    assert '"upper_floor_truth_corridor_probe"' in f1_cap
    fast_transaction = block[block.index("fast_corridor_transaction = source in ("):]
    assert '"upper_floor_truth_corridor_probe"' in fast_transaction


def test_upper_floor_wrapper_disables_duplicate_synchronous_visualization():
    """12/13/14 rendering is detached and must not delay stair ownership."""
    upper = (MANAGER.parent /
             "second_floor_exploration_manager.py").read_text(
                 encoding="utf-8")
    constructed = upper.index("manager = BaselineExplorationManager()")
    run = upper.index("manager.run()", constructed)
    block = upper[constructed:run]
    assert "manager.auto_generate_visualization = False" in block
    assert block.index("manager.auto_generate_visualization = False") < run


def test_upper_floor_exit_progress_guard_uses_bounded_restage_window():
    upper = (MANAGER.parent /
             "second_floor_exploration_manager.py").read_text(encoding="utf-8")
    assert '"room_exit_progress_timeout_seconds": 10.0' in upper
    assert "restage completed in 5.2 s" in upper


def test_f3_stair_wait_post_settle_drift_gets_bounded_physical_correction():
    """run160's 5.95 mm settle drift must not strand a strict 4/4 run."""
    upper = UPPER_MANAGER.read_text(encoding="utf-8")
    start = upper.index("def _third_floor_return_to_stair_wait")
    end = upper.index(
        "def _quiesce_exploration_executor_for_f3_return", start)
    block = upper[start:end]
    assert "final_lip_reached_pose" in block
    assert "correction_eligible = bool(" in block
    assert "correction_sim_limit = rospy.Duration(2.5)" in block
    assert "time.monotonic() + 10.0" in block
    assert "self._publish_f3_world_command(" in block
    assert "post_settle_correction_attempted" in block
    assert "post_settle_correction_ros_sim_s" in block
    assert "post_settle_hysteresis_accepted" in block
    assert "final_distance <= stair_wait_handoff_tolerance + 0.03" in block
    assert "final_reached_pose_drift <= 0.05" in block


def test_f3_stair_wait_rejection_is_not_mislabeled_as_room_exit_failure():
    """Room-contract and stair-wait failures are independent evidence."""
    upper = UPPER_MANAGER.read_text(encoding="utf-8")
    marker = 'rejection_reason = "physical_exit_requirement_not_met"'
    start = upper.index(marker)
    end = upper.index("# Publish the latched stair trigger", start)
    block = upper[start:end]
    assert "self._floor_number >= 3 and f3_stair_return_evidence" in block
    assert 'f3_stair_return_evidence.get("reason")' in block
    assert '"f3_stair_wait_return_failed"' in block
    assert "rejection_reason=rejection_reason" in block


def test_obstacle_view_route_faces_every_current_segment_not_next_segment():
    """Avoid run160 direct and run162 segmented lateral crawls."""
    manager = MANAGER.read_text(encoding="utf-8")
    start = manager.index(
        "obstacle_view_segment_heading = bool(")
    end = manager.index("# The deep obstacle EXIT is an L-shaped route", start)
    block = manager[start:end]
    assert 'room_role in ("G3", "G4")' in block
    assert "obstacle_room_route" in block
    assert "len(waypoints) <= 2" not in block
    assert "point[0]) - float(current[0])" in block
    assert "yaw = math.atan2(float(point[1]) - float(current[1])" in block
    dispatch = manager[manager.index("result = self._publish_waypoint(", end):]
    dispatch = dispatch[:dispatch.index("truth_relative_navigation=bool(")]
    assert "obstacle_view_segment_heading or" in dispatch


def test_deep_obstacle_exit_forces_every_audited_segment_heading():
    """run163 room4 EXIT must not execute audited legs laterally."""
    manager = MANAGER.read_text(encoding="utf-8")
    start = manager.index("deep_exit_segment_heading = bool(")
    end = manager.index("# EXIT has an explicit narrow-door", start)
    block = manager[start:end]
    assert "deep_truth_portal_exit" in block
    assert 'goal.get("deep_obstacle_front_gap_segmented_exit")' in block
    assert "point[0]) - float(current[0])" in block
    assert "if deep_exit_segment_heading:" in block
    assert "yaw = math.atan2(float(point[1]) - float(current[1])" in block
    dispatch = manager[manager.index("result = self._publish_waypoint(", end):]
    dispatch = dispatch[:dispatch.index("truth_relative_navigation=bool(")]
    assert "deep_exit_segment_heading or" in dispatch


def test_generated_truth_exit_crossing_prevents_drifted_door_retry():
    """run162's physical F3 EXIT must not be rejected by online door drift."""
    manager = MANAGER.read_text(encoding="utf-8")
    start = manager.index(
        'if goal.get("room_role") == "EXIT":',
        manager.index("generated_truth_entry_crossing_diagnostic"))
    end = manager.index("semantic_result = self.room_scheduler.record_result(",
                        start)
    audit = manager[start:end]
    assert "_room_exit_truth_crossing_confirmed(goal)" in audit
    assert 'goal["generated_truth_exit_crossing_confirmed"]' in audit
    assert "reported_success and truth_exit_ok" in audit
    assert "if reported_success and not truth_exit_ok" in audit
    core = CORE.read_text(encoding="utf-8")
    exit_branch = core[core.index('elif role == "EXIT":'):]
    exit_branch = exit_branch[:exit_branch.index("map_refresh_only =")]
    assert 'goal.get("generated_truth_exit_crossing_confirmed")' in exit_branch


def test_f3_strict_terminal_return_extends_to_safe_east_landing_only_after_exits():
    """Do not stop at ingress, and never activate the shortcut for partial F3."""
    baseline = MANAGER.read_text(encoding="utf-8")
    upper = UPPER_MANAGER.read_text(encoding="utf-8")
    start = baseline.index("def _stair_wait_target")
    end = baseline.index("def _stair_wait_reached", start)
    stair_target = baseline[start:end]
    assert "terminal_stair_wait_anchor_override" in stair_target
    assert "_mission_rooms_physically_exited()" in stair_target
    start = upper.index("def _seed_f3_terminal_stair_return_anchor")
    end = upper.index("def _physical_room_exit_evidence_count", start)
    seed = upper[start:end]
    assert 'upper_landing_east_clear' in seed
    assert 'truth_reference_pose' in seed
    assert 'activation_gate' in seed
    run = upper[upper.index("def _run_mission"):]
    assert "_seed_f3_terminal_stair_return_anchor(" in run
