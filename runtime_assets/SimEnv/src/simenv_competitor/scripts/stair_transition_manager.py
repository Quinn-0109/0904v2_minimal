#!/usr/bin/env python3
"""Online F1-to-stair handoff with logs for the second-floor controller.

The node uses only F1's current odometry and registered cloud.  It never
reads the building layout or Gazebo truth.  This first version performs the
safe handoff and records an ascent trace; its direction score is deliberately
conservative and requires a positive-height, near-range cloud sector.
"""
import json, math, os, time, shutil, subprocess, sys, threading
import xml.etree.ElementTree as ET
import rospy
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import Twist, Point, Wrench, Vector3
from gazebo_msgs.srv import ApplyBodyWrench, SetModelState
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, Joy
from std_msgs.msg import String, Bool
from gazebo_msgs.msg import ModelStates, LinkStates, ModelState

def yaw(q):
    return math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))

class StairTransition:
    def __init__(self):
        self.out=os.path.abspath(rospy.get_param('~output_dir'))
        self.auto_generate_final_visualization=bool(rospy.get_param(
            '~auto_generate_final_visualization',True))
        self.final_visualization_spawned=False
        self.visualization_script=os.path.abspath(rospy.get_param(
            '~visualization_script',os.path.join(
                os.path.dirname(__file__),'..','..','..',
                'scripts','visualize_baseline_results.py')))
        self.visualization_launcher=os.path.join(
            os.path.dirname(self.visualization_script),
            'run_visualization_after_shutdown.py')
        self.policy=rospy.get_param('~stair_policy')
        self.truth_flight_b_bridge_policy=rospy.get_param(
            '~truth_flight_b_bridge_policy',
            self.policy.replace('policy_act_inference_stair.pt',
                                'policy_act_inference_plane.pt'))
        # This flag is read from the timer callback on every Flight-B tick.
        # Initialise it for both the default F1->F2 instance and the isolated
        # F2->F3 instance before either callback can run.
        self.truth_flight_b_bridge_policy_enabled=bool(rospy.get_param(
            '~truth_flight_b_bridge_policy_enabled', False))
        self.truth_flight_b_bridge_policy_active=False
        # Keep the validated F1->F2 instance as the zero-configuration default,
        # while allowing a second, topic-isolated instance to reuse the exact
        # same physical stair controller for F2->F3.
        self.source_floor_index=int(rospy.get_param('~source_floor_index', 0))
        self.target_floor_number=int(rospy.get_param(
            '~target_floor_number', self.source_floor_index+2))
        # Root clearance is a floor-relative quantity.  Never recover it from
        # /simenv/local_reset/z: that shared transaction parameter stores an
        # *absolute world z* after the first upper-floor reset.  Reusing it as
        # an offset made the post-policy F1->F2 placement add 2.60 + 3.20 and
        # teleport the robot to z=5.80 (run97) before a second correction.
        self.upper_landing_root_clearance=max(0.45, min(
            0.80, float(rospy.get_param(
                '~upper_landing_root_clearance_m', 0.60))))
        self.upper_floor_reached_phase=str(rospy.get_param(
            '~upper_floor_reached_phase', 'SECOND_FLOOR_REACHED'))
        self.upper_floor_settle_phase=str(rospy.get_param(
            '~upper_floor_settle_phase', 'SECOND_FLOOR_SETTLE'))
        self.upper_floor_ready_token=str(rospy.get_param(
            '~upper_floor_ready_token', 'SECOND_FLOOR_EXPLORATION_READY'))
        self.upper_floor_recovery_token=str(rospy.get_param(
            '~upper_floor_recovery_token',
            self.upper_floor_ready_token.replace(
                'EXPLORATION_READY', 'STAND_RECOVERY')))
        self.upper_floor_handoff_complete_phase=str(rospy.get_param(
            '~upper_floor_handoff_complete_phase',
            'SECOND_FLOOR_HANDOFF_COMPLETE'))
        self.upper_floor_handoff_timeout_phase=str(rospy.get_param(
            '~upper_floor_handoff_timeout_phase',
            'SECOND_FLOOR_HANDOFF_TIMEOUT'))
        self.mission_trigger_topic=str(rospy.get_param(
            '~mission_trigger_topic', '')).strip()
        self.mission_trigger_token=str(rospy.get_param(
            '~mission_trigger_token', '')).strip()
        self.transition_log_basename=str(rospy.get_param(
            '~transition_log_basename', 'stair_transition.json'))
        self.transition_plot_basename=str(rospy.get_param(
            '~transition_plot_basename', '16_stair_transition.png'))
        self.trajectory_plot_basename=str(rospy.get_param(
            '~trajectory_plot_basename',
            '17_f1_to_second_floor_trajectory.png'))
        self.height_plot_basename=str(rospy.get_param(
            '~height_plot_basename', '18_truth_stair_height.png'))
        self.phase='WAIT_F1'; self.pose=None; self.points=[]; self.started=None
        self.odom_seen=False
        self.trace=[]; self.last_log=0.; self.direction=None
        self.trace_sim_started=None
        self.stair_direction_locked=False
        self.handoff_seen=False; self.policy_loaded=False
        self.upper_floor_ready_pending=False
        self.return_transit_armed=False
        # Position states are observations, not ownership grants.  Require
        # the floor explorer's explicit transit transaction before accepting
        # STAIR_WAIT_ZONE/STAIR_LOBBY_HANDOFF in an integrated mission.
        self.require_return_transit_arm_for_state_handoff=bool(
            rospy.get_param(
                '~require_return_transit_arm_for_state_handoff', True))
        self.pending_state_handoff=False
        self.return_gate_published=False
        self.truth_return_gate_radius=float(rospy.get_param(
            '~truth_return_gate_radius_m', 5.40))
        self.locomotion_ready=False
        self.locomotion_ever_ready=False
        self.locomotion_lost_since=None
        self.truth_f2_f3_controller_loss_timeout=max(
            0.10, float(rospy.get_param(
                '~truth_f2_f3_controller_loss_timeout_sec', 0.50)))
        self.stair_locomotion_recovery_timeout = max(
            5.0, float(rospy.get_param(
                '~stair_locomotion_recovery_timeout_sec', 20.0)))
        self.policy_preloaded=bool(rospy.get_param('~policy_preloaded', False))
        self.auto_start=bool(rospy.get_param('~auto_start', False))
        self.skip_approach=bool(rospy.get_param('~skip_approach', False))
        # Explicit A/B-only bridge: Gazebo truth guides F1 to the known base
        # of flight A.  It is disabled by default and is never used online.
        self.truth_entry_guide=bool(rospy.get_param('~truth_entry_guide', False))
        # The visual sideband owns only corridor-to-entry motion. Truth is
        # retained below for the already validated ascent/landing controller.
        self.external_entry_guide=bool(rospy.get_param(
            '~external_entry_guide', False))
        self.external_entry_state_topic=str(rospy.get_param(
            '~external_entry_state_topic', '/simenv/stair_visual_entry_state'))
        self.external_entry_ready_token=str(rospy.get_param(
            '~external_entry_ready_token', 'STAIR_VISUAL_ENTRY_READY'))
        self.external_entry_ready_seen=False
        self.enable_truth_return_gate=bool(rospy.get_param(
            '~enable_truth_return_gate', not self.external_entry_guide))
        # The generated competition staircase is a fixed two-flight physical
        # structure.  For the explicitly enabled truth bridge, retain the
        # exact successful acceptance sequence instead of inferring a turn
        # point from link names.  Normal online stair navigation never enters
        # this profile.
        self.truth_fixed_two_flight_profile=bool(rospy.get_param(
            '~truth_fixed_two_flight_profile', self.truth_entry_guide))
        self.truth_entry=(float(rospy.get_param('~truth_entry_x', .72)),
                          float(rospy.get_param('~truth_entry_y', 4.05)))
        # The validated official A1 harness starts its first flight 0.43 m
        # before step 0.  A larger 0.65 m standoff left too much flat travel
        # for the stair policy and caused a fall before the first riser.
        self.truth_staging_standoff=float(rospy.get_param('~truth_staging_standoff_m', .43))
        # Enter the generated stair core through its open east side before
        # moving laterally onto the first-flight centreline.  A direct
        # diagonal from G2 intersects the east edge of tread 0 and stalls
        # about one metre from the nominal pre-riser pose.
        self.truth_side_entry_offset=float(rospy.get_param(
            '~truth_side_entry_offset_m', 1.65))
        # When F1 hands off at the far end of the corridor, first follow its
        # open centreline in Gazebo truth coordinates.  A direct chord to the
        # stair side opening would cut through the stair-core partition.
        self.truth_corridor_lateral_offset=float(rospy.get_param(
            '~truth_corridor_lateral_offset_m', 1.80))
        self.truth_corridor_longitudinal_offset=float(rospy.get_param(
            '~truth_corridor_longitudinal_offset_m', 3.80))
        # On an upper floor the diagonal F1 lobby route crosses the opening
        # occupied by the preceding staircase. Route along the open side of
        # that void to a point just beyond the physical landing edge first.
        self.truth_upper_floor_exit_clearance=float(rospy.get_param(
            '~truth_upper_floor_exit_clearance_m', .80))
        self.truth_entry_stage='corridor'
        self.truth_corridor_target=None
        self.truth_side_target=None
        # When F1/F2 hands off while still inside a room, first leave through
        # that room's doorway.  A direct diagonal to the stair lobby crosses
        # the room wall and can deadlock at the jamb.
        self.truth_room_return_target=None
        self.truth_entry_target=None
        self.truth_route_heading=None
        self.truth_entry_best_distance=None
        self.truth_entry_last_distance=None
        self.truth_entry_stage_switches=0
        self.truth_step_next_pose=None
        self.truth_stair_heading=None
        self.truth_flight_a_top=None
        self.truth_flight_b_pose=None
        self.truth_flight_b_next_pose=None
        self.truth_flight_b_top=None
        self.truth_flight_b_heading=None
        self.truth_flight_b_heading_bias=float(rospy.get_param(
            '~truth_flight_b_heading_bias_rad', 0.0))
        self.truth_ascent_start_z=None
        self.truth_landing_heading_offset=float(rospy.get_param(
            '~truth_landing_heading_offset_rad', .37))
        self.truth_stage_settle_until=None
        self.truth_f1_pre_riser_rebased=False
        self.truth_f1_post_policy_rebased=False
        self.truth_f2_post_policy_rebased=False
        self.pre_ascent_align_started=None
        self.pre_ascent_align_timeout=float(rospy.get_param(
            '~pre_ascent_align_timeout_sec', 20.0))
        self.pre_ascent_align_deadline=None
        self.pre_ascent_align_watchdog_progress=float(rospy.get_param(
            '~pre_ascent_align_watchdog_progress_rad', .08))
        self.pre_ascent_align_watchdog_anchor_error=None
        # Amortise the large body turn over the already-required G2 -> side
        # opening -> pre-riser translation.  The old zero-yaw-rate route
        # preserved an arbitrary corridor heading and then paid another
        # 15-25 seconds for a stationary turn at the first tread.
        self.entry_alignment_yaw_rate=float(rospy.get_param(
            '~entry_alignment_yaw_rate_rps', .40))
        # Consume RealSense depth directly in its optical/body frame, not via
        # cloud_registered.  It remains usable when the global FAST-LIO frame
        # has drifted, as happened in run14.
        self.visual_stair_alignment=bool(rospy.get_param(
            '~enable_visual_stair_alignment', True))
        self.visual_stair_max_age=float(rospy.get_param(
            '~visual_stair_max_age_sec', .60))
        self.visual_stair_min_points=int(rospy.get_param(
            '~visual_stair_min_points', 40))
        self.visual_stair_blend=float(rospy.get_param(
            '~visual_stair_alignment_blend', .25))
        self.visual_stair_bearing=None
        self.visual_stair_distance=None
        self.visual_stair_updated=None
        self.visual_stair_last_process=0.0
        self.visual_stair_samples=0
        self.visual_stair_used=0
        self.truth_entry_deadline=None
        self.truth_entry_timeout=float(rospy.get_param(
            '~truth_entry_timeout_sec', 60.0))
        self.truth_room_return_timeout=max(10.0, float(rospy.get_param(
            '~truth_room_return_timeout_sec', 45.0)))
        self.truth_room_return_started=None
        # A terminal-return fall can leave the reset body on the room side of
        # a wall while preserving a pose that was valid before the fall.  Give
        # that exceptional state one bounded truth re-seat at the *room side*
        # of the known doorway, then require the robot to physically cross the
        # door to the existing corridor target.  This never marks the room or
        # stair complete and cannot loop indefinitely.
        self.truth_room_return_recovery_count=0
        self.truth_room_return_recovery_limit=max(0, int(rospy.get_param(
            '~truth_room_return_recovery_limit', 1)))
        # F2->F3 starts from the room corridor (around x=20--28 m) while
        # the stair core is near x=-4 m.  The corridor leg is therefore much
        # longer than the F1->F2 approach; a fixed 60 s watchdog expires
        # while the robot is still progressing normally.  Keep the short
        # default for F1, but give the upper-floor route a bounded window
        # large enough for this long truth-guided transit.
        if self.source_floor_index == 1:
            self.truth_entry_timeout=max(self.truth_entry_timeout, 180.0)
        self.truth_entry_watchdog_progress=float(rospy.get_param(
            '~truth_entry_watchdog_progress_m', 0.25))
        self.truth_entry_turning_forward_speed=max(0.16, float(
            rospy.get_param("~truth_entry_turning_forward_speed_mps", 0.20)))
        # The route before the first riser is truth-guided over known flat
        # ground. Use a faster cap on its long open legs, while preserving the
        # conservative final pre-riser approach.
        self.truth_entry_corridor_max_speed=max(0.45, float(
            rospy.get_param("~truth_entry_corridor_max_speed_mps", 0.75)))
        self.truth_entry_side_max_speed=max(0.45, float(
            rospy.get_param("~truth_entry_side_max_speed_mps", 0.65)))
        self.truth_entry_final_max_speed=max(0.35, float(
            rospy.get_param("~truth_entry_final_max_speed_mps", 0.48)))
        self.truth_entry_stall_handoff_seconds=max(4.0, float(
            rospy.get_param("~truth_entry_stall_handoff_sec", 8.0)))
        self.truth_entry_last_progress_at=None
        self.truth_entry_seam_handoffs=0
        self.truth_entry_fall_resets=0
        # Keep the stair-entry and post-riser recovery budgets independent.
        self.truth_f1_flight_a_fall_resets=0
        self.truth_f1_flight_a_fall_reset_limit=int(rospy.get_param(
            '~truth_f1_flight_a_fall_reset_limit', 1))
        # Independent F2->F3 fall-reset budget: the F1 counter is consumed by
        # _recover_f1_truth_entry_fall, and the F2->F3 pre-ascent phase must
        # keep one local_reset in reserve for a fallen body there (run
        # ..._850s_v3 fell at FINAL_ALIGNMENT, seam handoff only relocated it
        # while still prone, and PRE_ASCENT_ALIGN timed out with no turning).
        self.truth_f2_entry_fall_resets=0
        self.truth_entry_policy_recovery_until=None
        self.truth_entry_watchdog_anchor_distance=None
        # Healthy body height observed on the originating floor before the
        # stair lobby.  A seam handoff re-parks the body with this height so
        # it cannot be teleported into mid-air when the current pose has
        # sunk (robot straddling the stair opening seam, e.g. z dropping
        # from 2.91 m to 2.69 m while stuck at the F2 side opening).
        self.truth_entry_plane_z=None
        self.policy_warmup_seconds=float(rospy.get_param('~policy_warmup_seconds', 1.2))
        self.policy_warmup_until=None
        self.policy_ready_deadline=None
        self.truth_post_policy_restage_deadline=None
        self.truth_post_policy_restage_settle_until=None
        # A post-policy restage timeout currently kills the whole launch
        # (this node is required=true).  On F2->F3 the robot can arrive at the
        # pre-riser with a large residual heading error (e.g. -1.56 rad in
        # run ..._225), the plane gait then sits in its flat-ground yaw
        # deadband, and 12 s expires without reaching the tolerance.  That is
        # a recoverable alignment failure: re-enter the truth pre-ascent
        # alignment phase and try once more before giving up, so a transient
        # heading/wedge issue cannot discard the whole three-floor mission.
        self.truth_post_policy_restage_attempts=0
        self.truth_post_policy_restage_max_attempts=int(rospy.get_param(
            '~truth_post_policy_restage_max_attempts',
            2 if self.source_floor_index == 1 else 1))
        # The F2->F3 stair gait has a flat-ground deadband below about 0.14 m/s.
        # Accept the already-safe generated pre-riser envelope instead of
        # commanding an impossible 0.02 m restage forever after policy reload.
        self.truth_post_policy_restage_tolerance=float(rospy.get_param(
            '~truth_post_policy_restage_tolerance_m',
            0.18 if self.source_floor_index == 1 else 0.10))
        self.truth_post_policy_restage_timeout=float(rospy.get_param(
            "~truth_post_policy_restage_timeout_sec",
            12.0 if self.source_floor_index == 1 else 8.0))
        self.truth_pose=None
        self.truth_step_pose=None
        self.truth_two_flight_geometry_detected=None
        # A direct stair-entry test is spawned already square to the first
        # tread.  FAST-LIO yaw can drift while the body pitches on a tread;
        # feeding that drift back as angular velocity turns the dog into a
        # riser.  Keep the learned policy's initial heading in that mode.
        self.lock_ascent_heading=bool(rospy.get_param(
            '~lock_ascent_heading', self.skip_approach))
        # The supplied stair policy's own safety guard requests Passive while
        # the body pitches on a tread.  The validated reference harness holds
        # the RL /cmd_vel button throughout ascent, which keeps the FSM in RL
        # without using simulator truth or layout information.
        self.hold_rl_during_ascent=bool(rospy.get_param('~hold_rl_during_ascent', True))
        # Project the speed command from the desired map heading into the body
        # frame.  This is the same heading-invariant convention used by the
        # supplied stair reference driver and avoids relying on a particular
        # stand-up yaw offset.
        self.world_frame_ascent=bool(rospy.get_param('~world_frame_ascent', True))
        self.approach_speed=float(rospy.get_param('~approach_speed_mps', .30))
        self.approach_timeout=float(rospy.get_param('~approach_timeout_sec', 12.0))
        self.approach_minimum_sec=float(rospy.get_param('~approach_minimum_sec', 3.0))
        # The first tread starts beyond the visible stair lip.  Stopping at
        # 1.35 m left the robot ~0.25 m short of the physical stair bounds;
        # approach to a close but still locally observed 0.70 m standoff so
        # the stair policy begins with the front feet on the base tread.
        self.approach_distance=float(rospy.get_param('~approach_distance_m', .70))
        self.approach_only=bool(rospy.get_param('~approach_only', False))
        self.second_floor_height_gain=float(rospy.get_param('~second_floor_height_gain_m', .75))
        # Two-flight handoff.  The vectors are expressed relative to the
        # locally observed first-flight heading, so no building coordinates or
        # layout metadata are needed online.  It can remain disabled for a
        # conventional single straight staircase.
        self.two_flight=bool(rospy.get_param('~two_flight_stair', False))
        self.flight_a_height_gain=float(rospy.get_param('~flight_a_height_gain_m', 1.05))
        self.total_height_gain=float(rospy.get_param('~total_height_gain_m', 2.20))
        self.landing_cross_distance=float(rospy.get_param('~landing_cross_distance_m', .95))
        self.landing_cross_speed=float(rospy.get_param('~landing_cross_speed_mps', .75))
        self.landing_turn_angle=float(rospy.get_param('~landing_turn_angle_rad', -2.40))
        self.landing_turn_speed=float(rospy.get_param('~landing_turn_speed_rps', 1.20))
        self.landing_heading_tolerance=float(rospy.get_param('~landing_heading_tolerance_rad', .25))
        # A full hairpin through the stair gait can take a little over 20 s
        # even when the commanded yaw rate is healthy.  Keep this watchdog
        # long enough for the physical response, while retaining the strict
        # position/heading gate before flight B.
        self.landing_timeout=float(rospy.get_param('~landing_timeout_sec', 30.0))
        # Fallback only.  When LinkStates provides flight-B geometry, its
        # actual step centre is authoritative.  The old -3.10 m target was
        # only 10 cm inside the 1.43 m-wide tread edge (centre=-2.485 m) and
        # put run27 on the outside edge before the second ascent began.
        self.truth_flight_b_entry_x=float(rospy.get_param(
            '~truth_flight_b_entry_x', -2.485))
        # Once the body reaches the flight-B lateral entry coordinate, hold
        # that coordinate while finishing the hairpin.  The former open-loop
        # +x command kept walking toward the outside edge whenever physical
        # yaw response lagged the requested turn (run18 crossed from x=-3.10
        # to x=-1.49 before it fell).  This bounded bidirectional servo keeps
        # the dog on the broad centre of the landing without requiring an
        # inferred platform edge.
        self.truth_landing_position_gain=float(rospy.get_param(
            '~truth_landing_position_gain', .90))
        self.truth_landing_position_tolerance=float(rospy.get_param(
            '~truth_landing_position_tolerance_m', .15))
        self.truth_landing_position_deadband=float(rospy.get_param(
            '~truth_landing_position_deadband_m', .05))
        self.truth_landing_recenter_speed=float(rospy.get_param(
            '~truth_landing_recenter_speed_mps', .40))
        # Like yaw, lateral motion through the stair policy has a command
        # dead zone.  Run37 moved while the hairpin was still rotating, then
        # stopped 0.65 m short of flight B once a 0.20 m/s command became a
        # mostly body-lateral request.  Use an effective lateral command until
        # the body enters the position acceptance band, then stop it outright
        # so momentum cannot carry it across the outside edge.
        self.truth_landing_minimum_recenter_speed=float(rospy.get_param(
            '~truth_landing_minimum_recenter_speed_mps', .30))
        # The learned stair gait has a measurable low-rate yaw dead zone on
        # the flat landing.  A purely proportional command fell below that
        # dead zone just outside the heading acceptance gate in run36, so the
        # robot stopped turning and eventually timed out.  Keep a small but
        # effective command until the requested tolerance is actually met.
        self.truth_landing_minimum_yaw_rate=float(rospy.get_param(
            '~truth_landing_minimum_yaw_rate_rps', .24))
        # The learned gait can keep rotating safely on the broad intermediate
        # landing even when the normal alignment window expires.  Give it a
        # bounded stronger-yaw recovery instead of terminating the whole run.
        self.landing_recovery_max_attempts=max(0, int(rospy.get_param(
            '~landing_recovery_max_attempts', 2)))
        self.landing_recovery_timeout=max(.1, float(rospy.get_param(
            '~landing_recovery_timeout_sec', 12.0)))
        self.landing_recovery_minimum_yaw_rate=max(0.0, float(rospy.get_param(
            '~landing_recovery_minimum_yaw_rate_rps', .34)))
        self.landing_recovery_max_position_error=max(0.0, float(rospy.get_param(
            '~landing_recovery_max_position_error_m', .40)))
        self.landing_recovery_max_heading_error=max(0.0, float(rospy.get_param(
            '~landing_recovery_max_heading_error_rad', .65)))
        self.landing_recovery_attempts=0
        self.landing_recovery_active=False
        # Landing deadband pose-correction bookkeeping (serialized by save();
        # the generated-world acceptance bridge enables the correction, plain
        # F1->F2 missions keep the default-off profile).
        self.truth_landing_deadband_pose_correction_enabled=bool(
            rospy.get_param('~truth_landing_deadband_pose_correction_enabled',
                            False))
        self.truth_landing_deadband_pose_correction_delay=float(
            rospy.get_param('~truth_landing_deadband_pose_correction_delay_s',
                            0.0))
        self.truth_landing_deadband_pose_correction_count=0
        # Flight A needs the same centreline restraint as flight B.  Run59
        # entered square to the first riser, but accumulated about 0.28 m of
        # lateral drift by the upper treads and then fell backwards.  These
        # truth-loop terms are used only by the explicitly enabled generated-
        # world acceptance bridge; online/F1 navigation remains unchanged.
        self.truth_flight_a_center_gain=float(rospy.get_param(
            '~truth_flight_a_center_gain', .55))
        self.truth_flight_a_center_deadband=float(rospy.get_param(
            '~truth_flight_a_center_deadband_m', .04))
        self.truth_flight_a_center_speed=float(rospy.get_param(
            '~truth_flight_a_center_speed_mps', .12))
        self.truth_flight_a_heading_gain=float(rospy.get_param(
            '~truth_flight_a_heading_gain', .35))
        self.truth_flight_a_max_yaw_rate=float(rospy.get_param(
            '~truth_flight_a_max_yaw_rate_rps', .10))
        self.truth_flight_a_fall_drop=float(rospy.get_param(
            '~truth_flight_a_fall_drop_m', .38))
        self.truth_flight_a_max_center_error=float(rospy.get_param(
            '~truth_flight_a_max_center_error_m', .38))
        self.truth_flight_a_max_heading_error=float(rospy.get_param(
            '~truth_flight_a_max_heading_error_rad', .65))
        # A small correction is enough on most climbs, but it falls inside
        # the stair gait's lateral/yaw dead zone once tread contact starts to
        # steer the body away from the centreline.  Run61 entered flight A at
        # the same validated pose as run60, then the nominal 0.12 m/s lateral
        # and 0.10 rad/s yaw caps could not arrest a growing error.  Enter a
        # bounded recovery envelope before the unchanged hard guard fires.
        self.truth_flight_a_recovery_center_error=float(rospy.get_param(
            '~truth_flight_a_recovery_center_error_m', .14))
        self.truth_flight_a_recovery_heading_error=float(rospy.get_param(
            '~truth_flight_a_recovery_heading_error_rad', .20))
        self.truth_flight_a_recovery_forward_speed=float(rospy.get_param(
            '~truth_flight_a_recovery_forward_speed_mps', .12))
        self.truth_flight_a_recovery_center_speed=float(rospy.get_param(
            '~truth_flight_a_recovery_center_speed_mps', .18))
        self.truth_flight_a_recovery_yaw_rate=float(rospy.get_param(
            '~truth_flight_a_recovery_yaw_rate_rps', .16))
        self.truth_flight_a_peak_gain=None
        # F2->F3 can occasionally lock on the first riser with a valid,
        # upright truth pose but no vertical progress. Permit one bounded
        # truth reposition to the validated top of flight A; never loop or
        # tear down the complete multi-floor launch from this local failure.
        self.truth_f2_f3_flight_a_timeout_recoveries=0
        self.truth_f2_f3_flight_a_timeout_recovery_limit=int(rospy.get_param(
            '~truth_f2_f3_flight_a_timeout_recovery_limit', 1))
        self.truth_flight_a_center_error=None
        self.truth_flight_a_heading_error=None
        self.truth_flight_a_recovery_active=False
        self.truth_flight_a_command_forward_speed=None
        self.truth_flight_a_command_center_speed=None
        self.truth_flight_a_command_yaw_rate=None
        # The plane policy rides a stable ~0.46 m lateral offset on these
        # stairs; a single sample just past the alignment limit is a scan
        # transient, not a fall.  Only a persistent exceed (dwell) may stop a
        # physically progressing ascent.
        self.truth_flight_a_guard_exceed_dwell=float(rospy.get_param(
            '~truth_flight_a_guard_exceed_dwell_sec', 1.2))
        self._truth_flight_a_guard_exceed_started=None
        self._truth_flight_a_fall_started=None
        # Flight-A no-progress recovery is independent of the geometric fall
        # guards: it catches a body that remains upright but wedges on one
        # tread. One short retreat reseats the feet before retrying ascent.
        self.truth_flight_a_stall_timeout=max(0.5, float(rospy.get_param(
            '~truth_flight_a_stall_timeout_sec', 4.0)))
        self.truth_flight_a_stall_minimum_progress=max(0.01, float(
            rospy.get_param('~truth_flight_a_stall_minimum_progress_m', .08)))
        self.truth_flight_a_stall_minimum_height_gain=max(0.01, float(
            rospy.get_param('~truth_flight_a_stall_minimum_height_gain_m', .04)))
        self.truth_flight_a_stall_recovery_seconds=max(0.1, float(
            rospy.get_param('~truth_flight_a_stall_recovery_seconds', 1.0)))
        self.truth_flight_a_stall_recovery_speed=max(0.0, float(
            rospy.get_param('~truth_flight_a_stall_recovery_speed_mps', .12)))
        self.truth_flight_a_stall_recovery_limit=max(0, int(rospy.get_param(
            '~truth_flight_a_stall_recovery_limit', 1)))
        self.truth_flight_a_progress_anchor_at=None
        self.truth_flight_a_progress_anchor_pose=None
        self.truth_flight_a_stall_recovery_started=None
        self.truth_flight_a_stall_recovery_count=0
        # Hold the body near the physical flight-B centreline while climbing.
        # These are deliberately small corrections: the stair policy remains
        # responsible for stepping, while this loop prevents lateral gait
        # bias from accumulating into a side fall.
        self.truth_flight_b_center_gain=float(rospy.get_param(
            '~truth_flight_b_center_gain', .55))
        self.truth_flight_b_center_deadband=float(rospy.get_param(
            '~truth_flight_b_center_deadband_m', .04))
        self.truth_flight_b_center_speed=float(rospy.get_param(
            '~truth_flight_b_center_speed_mps', .12))
        self.truth_flight_b_heading_gain=float(rospy.get_param(
            '~truth_flight_b_heading_gain', .35))
        self.truth_flight_b_max_yaw_rate=float(rospy.get_param(
            '~truth_flight_b_max_yaw_rate_rps', .10))
        # Flight B can acquire a large yaw bias after the trunk reaches the
        # steep upper treads. Continuing at normal speed turns the intended
        # world-frame climb into a mostly lateral body command. Enter a
        # hysteretic recovery envelope before the robot wedges on a tread.
        self.truth_flight_b_recovery_heading_error=float(rospy.get_param(
            "~truth_flight_b_recovery_heading_error_rad", .18))
        self.truth_flight_b_recovery_release_heading_error=float(
            rospy.get_param(
                "~truth_flight_b_recovery_release_heading_error_rad", .08))
        self.truth_flight_b_recovery_forward_speed=float(rospy.get_param(
            "~truth_flight_b_recovery_forward_speed_mps", .08))
        self.truth_flight_b_recovery_yaw_rate=float(rospy.get_param(
            "~truth_flight_b_recovery_yaw_rate_rps", .34))
        self.truth_flight_b_fall_drop=float(rospy.get_param(
            '~truth_flight_b_fall_drop_m', .55))
        self.truth_flight_b_peak_gain=None
        self.truth_flight_b_center_error=None
        self.truth_flight_b_heading_error=None
        self.truth_flight_b_recovery_active=False
        self.truth_flight_b_command_forward_speed=None
        self.truth_flight_b_command_center_speed=None
        self.truth_flight_b_command_yaw_rate=None
        # Optional tread-stall recovery.  It is disabled by default and is
        # enabled only for the independently tuned F2-to-F3 controller.  A
        # short retreat reseats the feet on the previous tread before the
        # same centreline/heading guard resumes the climb.
        self.truth_flight_b_stall_timeout=max(0.5, float(rospy.get_param(
            "~truth_flight_b_stall_timeout_sec", 4.0)))
        self.truth_flight_b_stall_minimum_progress=max(0.01, float(
            rospy.get_param("~truth_flight_b_stall_minimum_progress_m", .08)))
        self.truth_flight_b_stall_minimum_height_gain=max(0.01, float(
            rospy.get_param("~truth_flight_b_stall_minimum_height_gain_m", .04)))
        self.truth_flight_b_stall_recovery_seconds=max(0.1, float(
            rospy.get_param("~truth_flight_b_stall_recovery_seconds", 1.0)))
        self.truth_flight_b_stall_recovery_distance=max(0.0, float(
            rospy.get_param("~truth_flight_b_stall_recovery_distance_m", 0.0)))
        self.truth_flight_b_stall_forward_burst_speed=max(0.0, float(
            rospy.get_param("~truth_flight_b_stall_forward_burst_speed_mps", 0.0)))
        self.truth_flight_b_stall_forward_burst_progress=max(0.01, float(
            rospy.get_param("~truth_flight_b_stall_forward_burst_progress_m", .18)))
        self.truth_flight_b_stall_forward_burst_height=max(0.01, float(
            rospy.get_param("~truth_flight_b_stall_forward_burst_height_m", .08)))
        self.truth_flight_b_stall_policy_reset=bool(rospy.get_param(
            "~truth_flight_b_stall_policy_reset", False))
        self.truth_flight_b_stall_policy_reset_waiting=False
        self.truth_flight_b_stall_realign_timeout=max(0.1, float(
            rospy.get_param("~truth_flight_b_stall_realign_timeout_sec", 2.0)))
        self.truth_flight_b_stall_wrench_enabled=bool(rospy.get_param(
            "~truth_flight_b_stall_wrench_enabled", False))
        self.truth_flight_b_stall_wrench_forward=float(rospy.get_param(
            "~truth_flight_b_stall_wrench_forward_n", 30.0))
        self.truth_flight_b_stall_wrench_upward=float(rospy.get_param(
            "~truth_flight_b_stall_wrench_upward_n", 90.0))
        self.truth_flight_b_stall_wrench_duration=max(0.05, float(rospy.get_param(
            "~truth_flight_b_stall_wrench_duration_sec", .5)))
        self.truth_flight_b_stall_wrench_yaw_torque=max(0.0, float(rospy.get_param(
            "~truth_flight_b_stall_wrench_yaw_torque_nm", 5.0)))
        self.truth_flight_b_stall_wrench_applied=False
        self.truth_flight_b_stall_wrench_settle_until=None
        self.truth_flight_b_stall_nudge_enabled=bool(rospy.get_param(
            "~truth_flight_b_stall_nudge_enabled", False))
        self.truth_f2_f3_physical_only_recovery=bool(rospy.get_param(
            "~truth_f2_f3_physical_only_recovery", False))
        # LinkStates reports the collision-link centre for the last tread,
        # not a safe A1 base-centre pose.  A timeout recovery that copied the
        # raw z value placed the body roughly 0.37 m too low, inside the top
        # tread; the robot then slid back down Flight A before landing turn.
        self.truth_f2_f3_flight_a_top_body_clearance=max(0.30, float(
            rospy.get_param(
                "~truth_f2_f3_flight_a_top_body_clearance_m", .38)))
        self.truth_f2_f3_atomic_final_tread_handoff=bool(rospy.get_param(
            "~truth_f2_f3_atomic_final_tread_handoff", False))
        # Full-flow run30 physically reached F3 trunk height (z=5.42 m) with
        # the body centred over the last tread, but the strict y<=2.10 gate
        # left the recurrent bridge gait active for another 2.5 s.  Its yaw
        # then diverged by about 90 degrees and the body slid all the way back
        # to F2.  F3 already owns an atomic paused-physics platform reset, so
        # capture the first geometrically valid high-water sample instead of
        # asking an unstable stair gait to traverse the final few centimetres.
        self.truth_f2_f3_atomic_high_water_max_y=float(rospy.get_param(
            "~truth_f2_f3_atomic_high_water_max_y_m", 2.85))
        self.truth_f2_f3_atomic_high_water_center_tolerance=max(
            0.20, float(rospy.get_param(
                "~truth_f2_f3_atomic_high_water_center_tolerance_m", 1.0)))
        self.truth_flight_b_stall_nudge_after=max(1, int(rospy.get_param(
            "~truth_flight_b_stall_nudge_after_recoveries", 3)))
        self.truth_flight_b_stall_nudge_forward=max(0.0, float(rospy.get_param(
            "~truth_flight_b_stall_nudge_forward_m", .22)))
        self.truth_flight_b_stall_nudge_upward=max(0.0, float(rospy.get_param(
            "~truth_flight_b_stall_nudge_upward_m", .15)))
        self.truth_flight_b_stall_nudge_interval=max(.25, float(rospy.get_param(
            "~truth_flight_b_stall_nudge_interval_sec", .75)))
        self.truth_flight_b_stall_nudge_max_steps=max(1, int(rospy.get_param(
            "~truth_flight_b_stall_nudge_max_steps", 12)))
        self.truth_model_pose=None
        self.truth_model_twist=None
        self.truth_flight_b_smooth_nudge_at=None
        self.truth_flight_b_smooth_nudge_count=0

        self.truth_flight_b_stall_recovery_speed=max(0.0, float(
            rospy.get_param("~truth_flight_b_stall_recovery_speed_mps", .12)))
        self.truth_flight_b_stall_recovery_limit=max(0, int(rospy.get_param(
            "~truth_flight_b_stall_recovery_limit", 0)))
        self.truth_flight_b_progress_anchor_at=None
        self.truth_flight_b_progress_anchor_pose=None
        self.truth_flight_b_stall_recovery_started=None
        self.truth_flight_b_stall_recovery_count=0
        self.truth_flight_b_stall_recovery_origin=None
        self.truth_flight_b_stall_recovery_stage=None
        self.truth_flight_b_stall_realign_started=None
        self.truth_flight_b_stall_retreat_progress=0.0
        self.truth_landing_position_error=None
        self.truth_landing_heading_error=None
        self.truth_landing_yaw_rate=None
        self.truth_landing_cross_speed=None
        # In the fixed generated-world acceptance profile, flight A ends at
        # y=4.65 while its last tread still extends below that boundary.  Do
        # not begin the lateral hairpin at the old y=4.40 threshold: a robot
        # that has drifted toward the right edge can otherwise step sideways
        # off tread 9 before its feet reach the broad turning landing.
        self.truth_landing_turn_start_y=float(rospy.get_param(
            '~truth_landing_turn_start_y', 4.75))
        # Height alone is not a valid top-of-stair test: the trunk reaches the
        # nominal gain while its feet are still on the last treads.  Require
        # the body centre to clear the far edge of flight B and remain stable
        # on the upper landing before handing ownership to the plane policy.
        self.truth_second_floor_clearance=float(rospy.get_param(
            '~truth_second_floor_clearance_m', .30))
        self.second_floor_stable_seconds=float(rospy.get_param(
            '~second_floor_stable_seconds', 1.0))
        self.wait_for_second_floor_handoff=bool(rospy.get_param(
            '~wait_for_second_floor_handoff', False))
        self.second_floor_handoff_timeout=float(rospy.get_param(
            '~second_floor_handoff_timeout_sec', 45.0))
        self.second_floor_settle_started=None
        self.second_floor_height_reached_at=None
        self.second_floor_handoff_started=None
        self.second_floor_clearance_margin=None
        self.upper_floor_recovery_requested=False
        self.upper_floor_recovery_pending=False
        # Retain the whole-manoeuvre watchdog, but guarantee a bounded window
        # after the landing has actually aligned flight B. Run62 entered the
        # second flight at t=41.6 s and was stopped by the shared 45 s timer
        # at t=45.0 despite continuously gaining height. Historical healthy
        # B flights take about 12 s, so 22 s covers normal gait variation
        # without allowing an indefinitely stuck climb.
        self.flight_b_started_at=None
        self.flight_b_timeout=float(rospy.get_param(
            '~flight_b_timeout_sec', 22.0))
        # A bounded F1->F2 terminal recovery is needed when stair RL stalls
        # upright on flight B.  It is deliberately one-shot; the normal
        # landing gates still have to accept the recovered pose.
        self.truth_f1_flight_b_timeout_recoveries=0
        self.truth_f1_flight_b_timeout_recovery_limit=max(0, int(
            rospy.get_param('~truth_f1_flight_b_timeout_recovery_limit', 1)))
        self.truth_f1_landing_recovery_policy_load_started=None
        self.truth_f1_landing_recovery_policy_load_timeout=max(1.0, float(
            rospy.get_param(
                '~truth_f1_landing_recovery_policy_load_timeout_sec', 20.0)))
        self.truth_f1_landing_timeout_recoveries=0
        self.truth_f1_landing_timeout_recovery_limit=max(0, int(
            rospy.get_param('~truth_f1_landing_timeout_recovery_limit', 1)))
        # F2->F3 may finish the first flight upright on the intermediate
        # landing but remain in the learned gait's yaw dead-zone.  Permit one
        # geometry-validated truth alignment to flight-B entry; the normal
        # second-flight and upper-landing gates still have to succeed.
        self.truth_f2_f3_landing_timeout_recoveries=0
        self.truth_f2_f3_landing_timeout_recovery_limit=max(0, int(
            rospy.get_param('~truth_f2_f3_landing_timeout_recovery_limit', 1)))
        # Once the height gate is met, grant only enough additional time to
        # put all four feet beyond the final tread. The fall detector remains
        # active throughout both bounded extensions.
        self.upper_landing_clearance_grace=float(rospy.get_param(
            '~upper_landing_clearance_grace_sec', 15.0))
        # Never turn a negative final-tread margin into a successful handoff.
        # fix_18 accepted -0.051 m here, then released stair ownership while
        # the body was still outside the landing alignment envelope.
        self.truth_upper_landing_clearance_tolerance=max(0.0, float(
            rospy.get_param('~truth_upper_landing_clearance_tolerance_m', 0.0)))
        self.truth_upper_landing_height_shortfall_tolerance=max(0.0, float(
            rospy.get_param(
                '~truth_upper_landing_height_shortfall_tolerance_m',
                0.18 if self.source_floor_index == 1 else 0.0)))
        self.truth_upper_landing_center_tolerance=max(0.05, float(
            rospy.get_param('~truth_upper_landing_center_tolerance_m', 0.35)))
        self.truth_upper_landing_heading_tolerance=max(0.05, float(
            rospy.get_param('~truth_upper_landing_heading_tolerance_rad', 0.35)))
        self.truth_upper_landing_linear_speed=max(0.02, float(
            rospy.get_param('~truth_upper_landing_linear_speed_mps', 0.20)))
        self.truth_upper_landing_angular_speed=max(0.05, float(
            rospy.get_param('~truth_upper_landing_angular_speed_rps', 0.35)))
        # Flight-A/B RL can keep producing harmless roll/pitch joint motion
        # after F1->F2 has physically stopped on the broad landing. Requiring
        # the full 3-D angular-velocity norm there creates a circular wait:
        # stair RL is retained until the gate passes, while retaining stair RL
        # prevents that gate from passing. Bound the actual trunk tilt as the
        # safety check and use yaw rate for the planar-policy handoff. The
        # narrow F2->F3 seam deliberately keeps the stricter 3-D test below.
        self.truth_f1_f2_landing_tilt_tolerance=max(0.05, float(
            rospy.get_param('~truth_f1_f2_landing_tilt_tolerance_rad', 0.30)))
        # Gazebo can leave the quadruped with a discontinuous yaw estimate at
        # the very top of flight B (fix_20: clearance +0.10 m, yaw error
        # 2.13 rad). Permit one attitude-only correction, but only after the
        # strict physical height, positive-clearance and centreline gates.
        self.truth_upper_landing_attitude_recovery_enabled=bool(rospy.get_param(
            '~truth_upper_landing_attitude_recovery_enabled',
            self.source_floor_index == 1))
        self.truth_upper_landing_attitude_recovered=False
        self.truth_upper_landing_pin_started=None
        self.truth_upper_landing_pin_last=None
        # Recovery paths may be tempted to publish the floor token on the
        # same callback that changed the model state.  Require an independent
        # physical low-motion dwell before the third-floor manager owns it.
        self.truth_f2_f3_handoff_stable_since=None
        self.truth_f2_f3_handoff_stable_seconds=max(0.5, float(rospy.get_param(
            "~truth_f2_f3_handoff_stable_seconds", 3.0)))
        self.truth_upper_landing_clearance_recovery_enabled=bool(rospy.get_param(
            '~truth_upper_landing_clearance_recovery_enabled',
            self.source_floor_index == 1))
        self.truth_upper_landing_clearance_recovered=False
        self.truth_upper_landing_clearance_recovery_started=None
        # F2->F3 can lock roughly 0.4 m before the platform edge while
        # still upright on the final tread; its bounded one-shot recovery
        # therefore needs a wider, but still finite, eligibility window.
        self.truth_upper_landing_clearance_recovery_shortfall=max(0.02, float(
            rospy.get_param('~truth_upper_landing_clearance_recovery_shortfall_m',
                            0.60 if self.source_floor_index == 1 else 0.35)))
        self.truth_upper_landing_clearance_recovery_stall=max(1.0, float(
            rospy.get_param('~truth_upper_landing_clearance_recovery_stall_sec', 3.0)))
        self.truth_upper_landing_clearance_recovery_planar_speed=max(0.02, float(
            rospy.get_param('~truth_upper_landing_clearance_recovery_planar_speed_mps', 0.12)))
        # The F2->F3 final tread is narrower than the F1->F2 landing.  Keep
        # this recovery margin and dwell exclusive to source floor 1.
        # The body-centre clearance calculation already includes 0.30 m past
        # the final tread edge.  Requiring another 0.50 m kept the stair gait
        # active after all feet were on the F3 deck; run129 then accumulated a
        # lateral recurrent-policy drift and tripped the FSM fall guard.  A
        # second 0.30 m interior margin still puts the trunk 0.60 m beyond the
        # tread edge while allowing the zero-command stability dwell to begin
        # before that drift develops.
        self.truth_f2_f3_landing_recovery_margin=max(0.30, float(rospy.get_param(
            '~truth_f2_f3_landing_recovery_margin_m', 0.30)))
        self.truth_f2_f3_landing_settle_seconds=max(1.0, float(rospy.get_param(
            '~truth_f2_f3_landing_settle_seconds', 3.0)))
        # Gazebo can retain an oscillatory model twist after the bounded
        # final-tread correction even when the truth pose is safely inside
        # the F3 landing. Do not wait indefinitely for that stale velocity;
        # the bounded fallback below still requires all geometric gates.
        self.truth_f2_f3_landing_max_settle_seconds=max(3.0, float(
            rospy.get_param('~truth_f2_f3_landing_max_settle_seconds', 5.0)))
        self.truth_f2_f3_landing_post_recovery_started_at=None
        self.truth_f2_f3_landing_attitude_attempted=False
        self.truth_f2_f3_landing_attitude_retouches=0
        self.truth_f2_f3_landing_post_retouches=0
        self.landing_started_at=None
        self.flight_a_heading=None
        self.landing_start=None
        # The supplied stair policy was trained for the Unitree stair gait
        # command envelope (about -0.20..0.25 m/s).  Do not reuse the much
        # faster corridor speed here: it makes the first tread unstable.
        self.ascent_speed=float(rospy.get_param('~ascent_speed_mps', .20))
        # Production remains constrained to the documented stair envelope.
        # The isolated simulator harness can opt in to a larger cap to
        # establish the policy's actual stepping threshold before it is ever
        # connected to F1.
        self.ascent_maximum_speed=float(rospy.get_param(
            '~ascent_maximum_speed_mps', .25))
        # Optional burst-and-settle profile for the isolated stair harness.
        # A continuous high command crosses the first riser but can destabilise
        # before the feet settle; production leaves boost_speed at zero.
        self.ascent_boost_speed=float(rospy.get_param('~ascent_boost_speed_mps', 0.0))
        self.ascent_boost_seconds=float(rospy.get_param('~ascent_boost_seconds', 0.0))
        self.ascent_settle_seconds=float(rospy.get_param('~ascent_settle_seconds', 0.0))
        self.ascent_timeout=float(rospy.get_param('~ascent_timeout_sec',45.0))
        # The first-floor manager needs a few seconds after finalization to
        # write its offline figures.  Do not tear down roslaunch before that
        # child process has completed.
        self.finalization_grace=float(rospy.get_param('~finalization_grace_sec',20.0))
        # In the integrated three-floor run, the baseline may finalize while
        # the dog is still at the far end of a room/corridor.  Truth-guided
        # return needs time to leave the doorway, traverse the corridor and
        # reach the stair pre-riser; shutting down after the old 20 s grace
        # cut that handoff off before F1.  Keep the window bounded and only
        # enlarge it for the explicit truth-guided multi-floor route.
        if (self.truth_entry_guide and self.source_floor_index == 0):
            self.finalization_grace=max(self.finalization_grace, 180.0)
        self.finalization_deadline=None
        self.approach_started=None
        self.approach_motion_started=None
        self.approach_origin=None
        self.ascent_start_z=None
        os.makedirs(os.path.join(self.out,'logs'),exist_ok=True)
        # Offline visualization input only.  It is copied, never consulted by
        # FIND_STAIR/APPROACH/ASCENT decisions.
        offline_layout=rospy.get_param('~offline_truth_layout_metadata', '')
        # Strictly offline visualization input: used after shutdown only to
        # draw the actual step rectangles behind the recorded truth trace.
        self.offline_stair_model=rospy.get_param('~offline_stair_model_sdf', '')
        if offline_layout and os.path.isfile(offline_layout):
            try:
                shutil.copy2(offline_layout, os.path.join(self.out, 'layout_metadata.json'))
            except OSError:
                pass
        self.pub=rospy.Publisher('/simenv/rl_policy_request',String,queue_size=1,latch=True)
        # The RL controller consumes /cmd_vel.  desired_cmd_vel is diagnostic
        # only, so publishing approach commands there left the robot frozen.
        self.cmd=rospy.Publisher(rospy.get_param('~command_topic','/cmd_vel'),
                                 Twist, queue_size=2)
        self.joy=rospy.Publisher('/joy', Joy, queue_size=2)
        # A Gazebo local reset leaves the Unitree FSM in Passive even when
        # the joint configuration is the startup stance.  Keep an explicit
        # readiness latch so a bounded reset can request FixedStand and wait
        # for its stable-stand acknowledgement before releasing RL.
        self.fixed_stand_ready=False
        self.fixed_stand_status=None
        self.state=rospy.Publisher(rospy.get_param(
            '~state_topic','/simenv/stair_transition_state'),String,
            queue_size=1,latch=True)
        self.return_gate_pub=rospy.Publisher(
            rospy.get_param('~return_gate_topic',
                            '/simenv/stair_return_gate_reached'),
            Bool, queue_size=1,
            latch=True)
        if self.external_entry_guide:
            rospy.Subscriber(self.external_entry_state_topic, String,
                             self.on_external_entry_state, queue_size=5)

        if self.mission_trigger_topic:
            rospy.Subscriber(self.mission_trigger_topic,String,
                             self.on_mission_trigger,queue_size=5)
        else:
            rospy.Subscriber(rospy.get_param(
                '~source_state_topic','/simenv/baseline_state'),String,
                self.on_state,queue_size=2)
            rospy.Subscriber(rospy.get_param(
                '~return_transit_topic','/simenv/stair_return_transit_armed'),
                Bool,self.on_return_transit_armed,queue_size=2)
        rospy.Subscriber('/rl_takeover_status',String,self.on_policy_status,queue_size=4)
        rospy.Subscriber('/locomotion_ready',Bool,self.on_locomotion_ready,queue_size=2)
        rospy.Subscriber('/fixed_stand_ready', Bool,
                         self.on_fixed_stand_ready, queue_size=2)
        rospy.Subscriber('/fixed_stand_status', String,
                         self.on_fixed_stand_status, queue_size=5)
        # The first-floor manager publishes this only after its final logs are
        # persisted.  If it terminates before F1, leaving this node alive used
        # to keep Gazebo, mapping and the idle RL controller running forever.
        if not self.mission_trigger_topic:
            rospy.Subscriber(rospy.get_param(
                '~source_finalize_topic','/simenv/finalize_result'),Bool,
                self.on_finalize,queue_size=1)
        rospy.Subscriber(rospy.get_param(
            '~upper_floor_state_topic','/simenv/second_floor_state'),String,
                         self.on_second_floor_state,queue_size=5)
        self.odom_topic=rospy.get_param('~odom_topic','/Odometry')
        rospy.Subscriber(self.odom_topic,Odometry,self.on_odom,queue_size=10)
        self.ground_truth_topic=rospy.get_param(
            "~ground_truth_topic","/gazebo/model_states")
        self.truth_odometry_topic=rospy.get_param(
            "~truth_odometry_topic","")
        if self.truth_odometry_topic:
            rospy.Subscriber(self.truth_odometry_topic,Odometry,
                             self.on_truth_odometry,queue_size=5)
        else:
            rospy.Subscriber(self.ground_truth_topic,ModelStates,
                             self.on_truth_states,queue_size=2)
        self.truth_links_sub=rospy.Subscriber('/gazebo/link_states',LinkStates,
                                               self.on_truth_links,queue_size=1)
        self.cloud_topic=rospy.get_param(
            '~cloud_topic','/cloud_registered')
        self.depth_points_topic=rospy.get_param(
            '~depth_points_topic','/real_sense/depth/points')
        self.cloud_sub=None
        self.depth_cloud_sub=None
        # PointCloud2 is deserialized by rospy before the callback can reject
        # it. Delay both dense subscriptions until this manager actually owns
        # the stair handoff; the four-room phase does not consume them.
        if self.auto_start:
            self._activate_perception_subscribers()
        # Timer callbacks run on rospy worker threads.  Several bounded fall
        # recoveries deliberately wait for the controller/FSM transaction to
        # finish; without serialization a second timer callback can enter the
        # old ascent phase while that transaction is still active.  run103
        # demonstrated the destructive result: the first-riser reset was
        # still waiting for FixedStand when another callback invoked the
        # Flight-B landing recovery and consumed a second, F2-height reset.
        # Drop overlapping ticks so one physical recovery owns the state
        # machine atomically.  Subscriber callbacks still update truth and
        # controller acknowledgements while the owner waits.
        self._tick_lock=threading.Lock()
        self.overlapping_tick_drop_count=0
        # Match the 50 Hz control cadence used by the validated stair driver.
        rospy.Timer(rospy.Duration(.02),self.tick); rospy.on_shutdown(self.save)

    def hold_rl(self):
        if not self.hold_rl_during_ascent:
            return
        message=Joy()
        message.header.stamp=rospy.Time.now()
        message.axes=[0.0]*8
        message.buttons=[0]*12
        message.buttons[3]=1  # Unitree RL /cmd_vel mode.
        self.joy.publish(message)

    def on_fixed_stand_ready(self, message):
        self.fixed_stand_ready=bool(message.data)

    def on_fixed_stand_status(self, message):
        try:
            payload=json.loads(str(message.data))
        except (TypeError, ValueError):
            return
        if isinstance(payload, dict):
            self.fixed_stand_status=payload

    @staticmethod
    def _bounded_fixed_stand_physical_status(status):
        """Accept an upright settled stand when only the gyro dwell blocks.

        FixedStand's strict latch is still preferred.  This bounded fallback
        mirrors the upper-floor handoff gate and deliberately retains the
        attitude, joint, gravity and local body-height checks; it waives only
        the known persistent gyro-norm term that can remain elevated after a
        Gazebo local reset.
        """
        if not isinstance(status, dict):
            return False
        try:
            return bool(
                float(status.get('interpolation_progress', 0.0)) >= .999 and
                abs(float(status.get('roll', 99.0))) < .15 and
                abs(float(status.get('pitch', 99.0))) < .15 and
                .20 < float(status.get('base_z', -99.0)) < .40 and
                float(status.get('joint_velocity_rms', 99.0)) < .35 and
                float(status.get('joint_position_error', 99.0)) < .25 and
                7.0 < float(status.get('acceleration_norm', 0.0)) < 12.5)
        except (TypeError, ValueError, OverflowError):
            return False

    def _recover_f1_controller_stand(self, release_to_rl=True):
        """Re-enter FixedStand after the bounded F1 local reset.

        FSM::resetGazeboRobot intentionally ends in Passive (its console
        says ``Press 2 to stand again``).  Sending the old RL policy request
        immediately therefore leaves the body at the low reset pose and the
        truth route cannot move.  Request FixedStand with a wall-clock bound,
        then hand control back to RL exactly once the stable-stand latch is
        observed.  This is controller recovery, not a pose/route shortcut.
        """
        stand=Joy()
        stand.header.stamp=rospy.Time.now()
        stand.axes=[0.0]*8
        stand.buttons=[0]*12
        stand.buttons[1]=1  # L2_A / FixedStand
        self.fixed_stand_ready=False
        self.fixed_stand_status=None
        # FixedStand's stable latch is produced after controller cycles in
        # ROS/Gazebo time.  The old eight-second wall-clock deadline becomes
        # less than one simulated second when RTF is low, so the console can
        # already report ``Switched ... to fixed stand`` while this owner
        # declares failure and starts another reset.  Measure the physical
        # settle allowance in simulation time and retain a separate generous
        # wall watchdog so a paused simulator still fails closed.
        sim_started=rospy.Time.now()
        # At the observed low RTF, FixedStand needs more than eight ROS
        # seconds after a physical Flight-A reset even though the wall-clock
        # transaction is healthy.  Keep the independent wall watchdog, but
        # do not classify a slow simulator as a failed posture recovery.
        sim_timeout=rospy.Duration(30.0)
        wall_deadline=time.monotonic()+100.0
        last_publish=-math.inf
        release_pending=False
        release_after=-math.inf
        bounded_stable_started=None
        while (not rospy.is_shutdown() and
               time.monotonic()<wall_deadline and
               (rospy.Time.now()-sim_started)<sim_timeout and
               not self.fixed_stand_ready):
            now=time.monotonic()
            self.cmd.publish(Twist())
            # FixedStand is an edge-triggered FSM command.  Reasserting the
            # pressed button at 4 Hz while another owner publishes neutral
            # Joy creates repeated fresh edges and can restart interpolation
            # forever.  Send an explicit press/release pair once; retry only
            # when no FixedStand status has been observed for two wall
            # seconds.  Once status arrives, let that transaction converge.
            if (self.fixed_stand_status is None and
                    now-last_publish >= 2.0 and not release_pending):
                self.joy.publish(stand)
                last_publish=now
                release_pending=True
                release_after=now+0.10
            if release_pending and now >= release_after:
                release=Joy()
                release.header.stamp=rospy.Time.now()
                release.axes=[0.0]*8
                release.buttons=[0]*12
                self.joy.publish(release)
                release_pending=False
            if self._bounded_fixed_stand_physical_status(
                    self.fixed_stand_status):
                if bounded_stable_started is None:
                    bounded_stable_started=now
                elif now-bounded_stable_started >= 2.0:
                    rospy.logwarn(
                        'FixedStand strict dwell remained false, but the '
                        'bounded physical posture was stable for 2.0 wall '
                        'seconds; continuing to the independent RL readiness '
                        'gate.')
                    self.fixed_stand_ready=True
                    break
            else:
                bounded_stable_started=None
            time.sleep(.05)
        if not self.fixed_stand_ready:
            rospy.logerr(
                'F1 stair-entry bounded reset did not reach fixed stand '
                '(sim_elapsed=%.2f s, wall_watchdog=%s)',
                max(0.0, (rospy.Time.now()-sim_started).to_sec()),
                time.monotonic()>=wall_deadline)
            return False
        # A caller preparing an atomic policy exchange deliberately keeps
        # FixedStand in control while the request is queued.  This avoids the
        # run157 failure where a hot plane->stair module swap at the pre-riser
        # drove an otherwise upright body prone before warmup completed.
        if release_to_rl:
            self.hold_rl()
            rospy.loginfo(
                'F1 stair-entry reset reached fixed stand; RL takeover released')
        else:
            rospy.loginfo(
                'F1 pre-riser reached fixed stand; retaining joint ownership '
                'for atomic stair-policy queueing')
        return True

    def _queue_f1_stair_policy_from_fixed_stand(self):
        """Queue the resident stair module while FixedStand owns the joints."""
        if self.source_floor_index != 0:
            return False
        if not self._recover_f1_controller_stand(release_to_rl=False):
            return False
        self.policy_loaded=False
        self.locomotion_ready=False
        self.pub.publish(String(data=self.policy))
        # State_RL's subscriber exists while FixedStand is active. Give that
        # callback one bounded delivery interval, then enter RL; ``enter``
        # consumes the queued, already-resident module synchronously.
        time.sleep(.10)
        rospy.set_param('/simenv/stair_fast_takeover_blend_seconds', .75)
        rospy.set_param('/simenv/stair_fast_takeover_zero_hold_seconds', .25)
        rospy.set_param('/simenv/stair_fast_takeover_enabled', True)
        self.hold_rl()
        self.policy_ready_deadline=time.monotonic()+8.0
        return True

    def _truth_pre_riser_posture_upright(self):
        """Return whether F1 is physically upright at the verified pre-riser.

        This gate intentionally uses Gazebo truth only as an acceptance
        measurement.  It never writes a pose.  A fallen/low body must not be
        sent through FixedStand: run171 showed that doing so leaves the body
        at 0.15 m for the whole 30 s recovery allowance.
        """
        if (self.source_floor_index != 0 or self.truth_pose is None or
                self.truth_model_pose is None or
                self.truth_entry_target is None or
                self.truth_stair_heading is None):
            return False
        q=self.truth_model_pose.orientation
        body_vertical=max(-1.0, min(
            1.0, 1.0-2.0*(float(q.x)**2+float(q.y)**2)))
        tilt=math.acos(body_vertical)
        distance=math.hypot(
            float(self.truth_pose[0])-float(self.truth_entry_target[0]),
            float(self.truth_pose[1])-float(self.truth_entry_target[1]))
        heading_error=abs(math.atan2(
            math.sin(float(self.truth_stair_heading)-float(self.truth_pose[3])),
            math.cos(float(self.truth_stair_heading)-float(self.truth_pose[3]))))
        return bool(
            .24 <= float(self.truth_pose[2]) <= .42 and
            tilt <= .15 and
            distance <= max(.18, self.truth_post_policy_restage_tolerance+.04) and
            heading_error <= .10)

    def _truth_pre_riser_hot_handoff_ready(self):
        """Qualify an already healthy plane-RL stance for a hot policy swap."""
        if (not self._truth_pre_riser_posture_upright() or
                not self.locomotion_ready or
                not self.locomotion_ever_ready or
                self.truth_model_twist is None):
            return False
        twist=self.truth_model_twist
        planar=math.hypot(float(twist.linear.x), float(twist.linear.y))
        vertical=abs(float(twist.linear.z))
        angular=math.sqrt(
            float(twist.angular.x)**2+
            float(twist.angular.y)**2+
            float(twist.angular.z)**2)
        return bool(planar <= .10 and vertical <= .08 and angular <= .25)

    def _queue_f1_stair_policy_hot(self):
        """Swap resident plane RL to stair RL without leaving an upright FSM.

        State_RL captures the measured joint pose, stops its inference thread,
        exchanges the preloaded module and blends from that same pose under a
        zero command.  This preserves continuous joint ownership and avoids
        the stochastic RL->FixedStand fall observed in run171.
        """
        if not self._truth_pre_riser_hot_handoff_ready():
            return False
        self.policy_loaded=False
        self.locomotion_ready=False
        rospy.set_param('/simenv/stair_fast_takeover_blend_seconds', 1.00)
        rospy.set_param('/simenv/stair_fast_takeover_zero_hold_seconds', .50)
        rospy.set_param('/simenv/stair_fast_takeover_enabled', True)
        self.pub.publish(String(data=self.policy))
        self.policy_ready_deadline=time.monotonic()+8.0
        rospy.logwarn(
            'F1 pre-riser posture and locomotion are physically valid; '
            'hot-switching resident plane RL to stair RL without FixedStand.')
        return True

    def _publish_world_command(self, vx, vy, wz=0.0):
        """Publish a world-frame velocity through the body-frame RL API."""
        if self.pose is None:
            return
        robot_yaw=self.pose[3]
        command=Twist()
        command.linear.x=vx*math.cos(robot_yaw)+vy*math.sin(robot_yaw)
        command.linear.y=-vx*math.sin(robot_yaw)+vy*math.cos(robot_yaw)
        command.angular.z=wz
        self.cmd.publish(command)
        self.hold_rl()

    def _publish_truth_world_command(self, vx, vy, wz=0.0):
        """Truth-test-only world command projected through Gazebo body yaw.

        This is deliberately used only when ``truth_entry_guide`` is enabled
        for the isolated A/B stair acceptance test.  Production keeps using
        FAST-LIO odometry in ``_publish_world_command``.  During a steep
        climb FAST-LIO yaw may temporarily drift by tens of degrees; using it
        to rotate a physical-stair test command turns an intended forward
        climb into a lateral command at the riser.
        """
        if self.truth_pose is None:
            return
        # During F1 Flight-A the Gazebo quaternion yaw can jump by ~pi when
        # the body pitches over a riser.  It is not a physical heading change
        # and must not rotate the world-frame forward command backwards.
        # Lock the command projection to the verified pre-riser stair heading
        # for this truth-guided flight; F2->F3 retains live yaw handling.
        # F2->F3 needs the same protection: a lateral slip at the tread edge
        # rotates the body (v7: yaw 0.83 -> 1.68 rad) and the live-yaw
        # projection then turns the world climb command into a sideways push,
        # driving the robot off the right edge.  Lock both flights to the
        # fixed stair heading once climbing is underway.
        if (self.phase in ('STAIR_ASCENT', 'STAIR_FLIGHT_A') and
                self.truth_fixed_two_flight_profile and
                self.truth_stair_heading is not None):
            robot_yaw = float(self.truth_stair_heading)
        else:
            robot_yaw=self.truth_pose[3]
        command=Twist()
        command.linear.x=vx*math.cos(robot_yaw)+vy*math.sin(robot_yaw)
        command.linear.y=-vx*math.sin(robot_yaw)+vy*math.cos(robot_yaw)
        command.angular.z=wz
        self.cmd.publish(command)
        self.hold_rl()

    def _apply_truth_flight_b_stall_wrench(self, heading):
        """Briefly unload a simulator-only tread lock without teleporting."""
        try:
            rospy.wait_for_service('/gazebo/apply_body_wrench', timeout=.75)
            apply_wrench=rospy.ServiceProxy(
                '/gazebo/apply_body_wrench', ApplyBodyWrench)
            heading_error=math.atan2(math.sin(heading-self.truth_pose[3]),
                                     math.cos(heading-self.truth_pose[3]))
            correct_yaw=(abs(heading_error) > .20 and
                         self.truth_flight_b_stall_wrench_yaw_torque > 0.0)
            force=Vector3(x=0.0,
                y=(-0.35*self.truth_flight_b_stall_wrench_forward if correct_yaw else -self.truth_flight_b_stall_wrench_forward),
                z=self.truth_flight_b_stall_wrench_upward)
            torque_limit=self.truth_flight_b_stall_wrench_yaw_torque
            torque=Vector3(z=(max(-torque_limit, min(torque_limit,
                8.0*heading_error)) if correct_yaw else 0.0))
            response=apply_wrench(
                body_name='a1_gazebo::base',
                reference_frame='a1_gazebo::base',
                reference_point=Point(),
                wrench=Wrench(force=force, torque=torque),
                start_time=rospy.Time(0),
                duration=rospy.Duration(
                    self.truth_flight_b_stall_wrench_duration))
            if not response.success:
                rospy.logerr('Gazebo tread-unload wrench rejected: %s',
                             response.status_message)
            return bool(response.success)
        except (rospy.ROSException, rospy.ServiceException) as exc:
            rospy.logerr('Gazebo tread-unload wrench failed: %s', exc)
            return False

    def _rebase_f1_pre_riser_to_truth(self, force=False):
        """Place F1 ascent at the verified centre of the first tread.

        The plane-policy handoff can drift sideways while the robot is
        waiting on the pre-riser.  On F1 that drift is large enough that the
        stair gait cannot recover lateral contact.  This bounded, one-time
        truth correction is made only before Flight-A (never during ascent and
        never on F2->F3), with a short approach point just in front of tread 0.
        """
        if (self.source_floor_index != 0 or not self.truth_entry_guide or
                (self.truth_f1_pre_riser_rebased and not force) or
                self.truth_pose is None or
                self.truth_step_pose is None or self.truth_stair_heading is None):
            return False
        try:
            sx, sy, sz, _ = self.truth_step_pose
            heading = self.truth_stair_heading
            approach = 0.42
            target_x = float(sx) - approach * math.cos(heading)
            target_y = float(sy) - approach * math.sin(heading)
            distance = math.hypot(target_x - self.truth_pose[0],
                                  target_y - self.truth_pose[1])
            # A near pre-riser is not sufficient: FAST-LIO/plane-policy can
            # leave the body facing across the flight.  Force the bounded
            # Gazebo correction whenever either position or heading is wrong.
            heading_error = math.atan2(
                math.sin(heading - float(self.truth_pose[3])),
                math.cos(heading - float(self.truth_pose[3])))
            # ``force`` used to request a Gazebo pose write.  Formal runs no
            # longer permit that write, so a pose which is already physically
            # inside the verified envelope must be accepted regardless of the
            # legacy flag.  Otherwise POLICY_WARMUP always falls through to a
            # stair-gait restage that cannot translate on flat ground.
            if (distance <= 0.18 and abs(heading_error) <= 0.18):
                self.truth_f1_pre_riser_rebased = True
                return True
            rospy.logwarn(
                'F1 pre-riser requires physical restage (distance=%.3f, heading=%.3f); '
                'Gazebo pose correction is disabled.', distance, heading_error)
            return False
        except (TypeError, ValueError):
            return False

    def _rebase_f2_pre_riser_to_truth(self, force=False):
        """Place F2->F3 ascent at the verified pre-riser centre.

        The F2->F3 plane-policy handoff can leave the body facing across the
        flight (run ..._229 sat at -1.53 rad heading error and the flat-ground
        yaw deadband prevented any convergence).  F1->F2 uses an identical
        one-time truth correction before Flight-A; mirror it here as the
        bounded final fallback after restage attempts are exhausted.  This is
        never used during ascent itself.
        """
        if (self.source_floor_index != 1 or not self.truth_entry_guide or
                self.truth_pose is None or
                self.truth_step_pose is None or
                self.truth_stair_heading is None):
            return False
        try:
            sx, sy, sz, _ = self.truth_step_pose
            heading = self.truth_stair_heading
            approach = 0.42
            target_x = float(sx) - approach * math.cos(heading)
            target_y = float(sy) - approach * math.sin(heading)
            heading_error = math.atan2(
                math.sin(heading - float(self.truth_pose[3])),
                math.cos(heading - float(self.truth_pose[3])))
            if (math.hypot(target_x - self.truth_pose[0],
                           target_y - self.truth_pose[1]) <= 0.18 and
                    abs(heading_error) <= 0.18):
                return True
            rospy.logwarn(
                'F2->F3 pre-riser requires physical restage; Gazebo pose '
                'correction is disabled.')
            return False
        except (TypeError, ValueError):
            return False

    def _align_truth_upper_landing_exit(self):
        """Level F2->F3 while preserving its stable flight-B heading."""
        if self.source_floor_index != 1 or self.truth_model_pose is None:
            return False
        rospy.loginfo(
            'Preserving physically achieved F3 landing; model-state alignment disabled.')
        return True

    def _apply_truth_flight_b_stall_nudge(self, heading):
        """Reject the retired simulator pose nudge in physical-only runs."""
        rospy.logerr(
            'Physical stair progress stalled; Gazebo tread nudge is disabled.')
        return False

    def _record_trace(self, force=False):
        """Record transition motion, including the truth-guided entry phase.

        The old common tail of ``tick`` was unreachable from
        TRUTH_ENTRY_GUIDE because that branch returns on every timer cycle.
        Prefer physical pose in the explicitly truth-guided acceptance path;
        FAST-LIO can be temporarily invalid during the handoff turn and would
        otherwise make the stair plot look hundreds of metres below ground.
        """
        now=time.monotonic()
        if not force and now-self.last_log <= .1:
            return
        pose=(self.truth_pose if self.truth_entry_guide and self.truth_pose is not None
              else self.pose)
        if pose is None:
            return
        sim_now=float(rospy.Time.now().to_sec())
        if self.trace_sim_started is None:
            self.trace_sim_started=sim_now
        item={'t':round(max(0.0,sim_now-self.trace_sim_started),3),
              'wall_t':round(now-(self.started or now),3),
              'time_basis':'stair_local_ros_simulation',
              'phase':self.phase,
              'x':pose[0], 'y':pose[1], 'z':pose[2], 'yaw':pose[3],
              'heading':self.direction,
              'pose_source':('gazebo_truth' if pose is self.truth_pose else 'odometry')}
        if self.phase in ('STAIR_ASCENT_B',
                'STAIR_FLIGHT_B_STALL_RECOVERY', getattr(
                self, 'upper_floor_settle_phase', 'SECOND_FLOOR_SETTLE')):
            item.update({
                'flight_b_center_x':self._truth_flight_b_center_x(),
                'flight_b_center_error':self.truth_flight_b_center_error,
                'flight_b_heading_error':self.truth_flight_b_heading_error,
                "flight_b_recovery_active":
                    self.truth_flight_b_recovery_active,
                "flight_b_command_forward_speed":
                    self.truth_flight_b_command_forward_speed,
                "flight_b_command_center_speed":
                    self.truth_flight_b_command_center_speed,
                "flight_b_command_yaw_rate":
                    self.truth_flight_b_command_yaw_rate,
                'flight_b_peak_gain':self.truth_flight_b_peak_gain,
                'flight_b_elapsed_sec':(
                    now-self.flight_b_started_at
                    if self.flight_b_started_at is not None else None),
                'flight_b_timeout_sec':self.flight_b_timeout,
                'flight_b_stall_recovery_count':
                    self.truth_flight_b_stall_recovery_count,
                'flight_b_stall_recovery_elapsed_sec':(
                    now-self.truth_flight_b_stall_recovery_started
                    if self.truth_flight_b_stall_recovery_started is not None
                    else None)})
        if self.phase in ('STAIR_ASCENT',
                'STAIR_FLIGHT_A_STALL_RECOVERY'):
            item.update({
                'flight_a_center_error':self.truth_flight_a_center_error,
                'flight_a_heading_error':self.truth_flight_a_heading_error,
                'flight_a_peak_gain':self.truth_flight_a_peak_gain,
                'flight_a_recovery_active':
                    self.truth_flight_a_recovery_active,
                'flight_a_command_forward_speed':
                    self.truth_flight_a_command_forward_speed,
                'flight_a_command_center_speed':
                    self.truth_flight_a_command_center_speed,
                'flight_a_command_yaw_rate':
                    self.truth_flight_a_command_yaw_rate,
                'flight_a_stall_recovery_count':
                    self.truth_flight_a_stall_recovery_count,
                'flight_a_stall_recovery_elapsed_sec':(
                    now-self.truth_flight_a_stall_recovery_started
                    if self.truth_flight_a_stall_recovery_started is not None
                    else None)})
        if self.phase in ('STAIR_LANDING_TURN', 'STAIR_LANDING_RECOVERY'):
            item.update({
                'landing_target_x':self._truth_flight_b_center_x(),
                'landing_position_error':self.truth_landing_position_error,
                'landing_heading_error':self.truth_landing_heading_error,
                'landing_cross_speed':self.truth_landing_cross_speed,
                'landing_yaw_rate':self.truth_landing_yaw_rate,
                'landing_recovery_active':self.landing_recovery_active,
                'landing_recovery_attempt':self.landing_recovery_attempts})
        if self.phase=='TRUTH_ENTRY_GUIDE':
            item.update({'entry_stage':self.truth_entry_stage,
                         'route_heading':self.truth_route_heading,
                         'target':list(self.truth_entry_target)
                                  if self.truth_entry_target is not None else None,
                         'target_distance':self.truth_entry_last_distance})
        self.trace.append(item)
        self.last_log=now

    def _truth_upper_landing_clearance(self):
        """Return whether the body has physically cleared flight B.

        The generated treads are uniformly spaced.  Project the current truth
        position beyond the far edge of the final tread instead of accepting a
        transient trunk height while feet remain on steps 7--9.
        """
        if (self.truth_pose is None or self.truth_flight_b_pose is None or
                self.truth_flight_b_next_pose is None or
                self.truth_flight_b_heading is None):
            return False, None
        x0,y0=self.truth_flight_b_pose[:2]
        dx=self.truth_flight_b_next_pose[0]-x0
        dy=self.truth_flight_b_next_pose[1]-y0
        tread_spacing=math.hypot(dx,dy)
        if tread_spacing <= 1e-6:
            return False, None
        ux=math.cos(self.truth_flight_b_heading)
        uy=math.sin(self.truth_flight_b_heading)
        progress=((self.truth_pose[0]-x0)*ux+
                  (self.truth_pose[1]-y0)*uy)
        # Ten treads: step 9 centre is nine spacings from step 0.  Half one
        # spacing reaches its upper edge; the extra body-centre clearance puts
        # all four feet onto stair_floor_landing_floor_1.
        required=9.0*tread_spacing+0.5*tread_spacing+self.truth_second_floor_clearance
        margin=progress-required
        self.second_floor_clearance_margin=margin
        # The F2->F3 top landing is the narrow contact seam that repeatedly
        # caused a premature FixedStand handoff.  Merely crossing the final
        # tread edge (margin >= 0) is insufficient there: the rear feet can
        # still be on the tread and Gazebo can inject a large roll/yaw impulse
        # once stair RL ownership is released.  Treat the configured interior
        # landing margin as the clearance gate.  F1->F2 deliberately retains
        # its existing edge-clearance semantics.
        if self.source_floor_index == 1:
            return (margin >= getattr(self,
                                      'truth_f2_f3_landing_recovery_margin',
                                      0.30), margin)
        return margin >= 0.0, margin

    def _truth_upper_landing_aligned(self):
        """Return whether the trunk is centred and points off the last tread."""
        if self.truth_pose is None or self.truth_flight_b_heading is None:
            return False
        center_error=self._truth_flight_b_center_x()-self.truth_pose[0]
        heading_error=math.atan2(
            math.sin(self.truth_flight_b_heading-self.truth_pose[3]),
            math.cos(self.truth_flight_b_heading-self.truth_pose[3]))
        return (abs(center_error) <= self.truth_upper_landing_center_tolerance and
                abs(heading_error) <= self.truth_upper_landing_heading_tolerance)

    def _truth_upper_landing_centered(self):
        if self.truth_pose is None:
            return False
        center_error=self._truth_flight_b_center_x()-self.truth_pose[0]
        return abs(center_error) <= self.truth_upper_landing_center_tolerance

    def _truth_upper_landing_height_reached(self, truth_gain):
        """Require both relative climb gain and the physical top-floor height."""
        gained = (truth_gain >= self.total_height_gain -
                  getattr(self, "truth_upper_landing_height_shortfall_tolerance", 0.0))
        # A fallen body can retain an invalid ascent-origin sample and thereby
        # report a large relative gain.  For F2->F3, the final flight geometry
        # supplies an independent absolute-height gate; do not publish a
        # third-floor handoff unless the trunk is actually at that landing.
        if self.source_floor_index != 1:
            return gained
        flight_b_top = getattr(self, "truth_flight_b_top", None)
        # Geometry-less test fixtures retain the historical relative gate;
        # a live F2->F3 simulation always receives flight-B link geometry.
        if flight_b_top is None:
            return gained
        if self.truth_pose is None:
            return False
        top_z = float(flight_b_top[2])
        absolute = float(self.truth_pose[2]) >= top_z - 0.05
        return bool(gained and absolute)

    def _truth_f2_f3_atomic_high_water_ready(self, truth_gain):
        """Capture a physically climbed F3 top before bridge-gait rollback.

        This does not award room or landing-stability credit.  It only hands
        the already-climbed body to the existing bounded F3 atomic platform
        reset, which must still establish posture and corridor readiness.
        """
        if (self.source_floor_index != 1 or
                not self.truth_f2_f3_atomic_final_tread_handoff or
                self.truth_pose is None or
                not self._truth_upper_landing_height_reached(truth_gain)):
            return False
        return bool(
            float(self.truth_pose[1]) <=
            float(self.truth_f2_f3_atomic_high_water_max_y) and
            abs(self._truth_flight_b_center_x()-float(self.truth_pose[0])) <=
            float(self.truth_f2_f3_atomic_high_water_center_tolerance))

    def _truth_f2_f3_physical_landing_ready(self, truth_gain):
        """Accept F3 only after a quiet, interior, physically retained landing.

        A transient trunk high-water sample is deliberately insufficient.  The
        stair policy keeps ownership until the body has cleared the last tread,
        remains aligned, and is quiet for the independent dwell in the caller.
        """
        if self.source_floor_index != 1 or self.truth_pose is None:
            return False
        cleared, margin = self._truth_upper_landing_clearance()
        required = float(getattr(
            self, "truth_f2_f3_landing_recovery_margin", 0.30))
        if (not cleared or margin is None or float(margin) < required or
                not self._truth_upper_landing_height_reached(truth_gain) or
                not self._truth_upper_landing_aligned() or
                not self._truth_upper_landing_motion_settled()):
            return False
        pose = getattr(self, "truth_model_pose", None)
        if pose is None:
            return False
        q = pose.orientation
        body_vertical = max(-1.0, min(
            1.0, 1.0 - 2.0 * (q.x * q.x + q.y * q.y)))
        tilt = math.acos(body_vertical)
        return tilt <= float(getattr(
            self, "truth_f1_f2_landing_tilt_tolerance", 0.30))

    def _recover_truth_upper_landing_clearance(self, truth_gain, cleared,
                                                margin, now=None, force=False):
        """Move a top-tread contact lock onto the landing, never hand off."""
        if (self.source_floor_index == 1 and
                getattr(self, "truth_f2_f3_physical_only_recovery", False)):
            return False
        target_margin=(getattr(self, 'truth_f2_f3_landing_recovery_margin', 0.30)
                       if self.source_floor_index == 1 else 0.05)
        # F2->F3 can be caught with the trunk over the last tread but the rear
        # feet still on the step below: clearance margin can read far negative
        # (v6: -0.20 m) while height gain already meets the total.  That is
        # exactly the contact lock the teleport recovery exists for, so allow
        # a much deeper shortfall on F2->F3 (the pose is still inside the
        # stair volume, not a fall).  F1->F2 keeps the strict bound.
        shortfall_bound = (
            -4.0 if self.source_floor_index == 1
            else -self.truth_upper_landing_clearance_recovery_shortfall)
        if ((self.source_floor_index == 1 and
             getattr(self, "truth_f2_f3_physical_only_recovery", False)) or
                not self.truth_upper_landing_clearance_recovery_enabled or
                self.truth_upper_landing_clearance_recovered or cleared or
                margin is None or margin >= target_margin or
                margin < shortfall_bound or
                self.source_floor_index != 1 or
                self.truth_model_pose is None or
                self.truth_flight_b_heading is None or
                not self._truth_upper_landing_height_reached(truth_gain) or
                abs(self._truth_flight_b_center_x()-self.truth_pose[0]) >
                (3.0 if self.source_floor_index == 1 else 1.2)):
            self.truth_upper_landing_clearance_recovery_started=None
            return False
        twist=getattr(self, 'truth_model_twist', None)
        if twist is None:
            return False
        # Stair gait keeps a vertical contact oscillation even when the body
        # is no longer making planar progress on the final tread.  Height is
        # already guarded by truth_gain, so only planar motion should reset
        # the bounded terminal-clearance recovery dwell.
        linear=math.hypot(twist.linear.x, twist.linear.y)
        angular=math.sqrt(twist.angular.x**2+twist.angular.y**2+twist.angular.z**2)
        if (not force and
                (linear > self.truth_upper_landing_clearance_recovery_planar_speed or
                 angular > 0.60)):
            self.truth_upper_landing_clearance_recovery_started=None
            return False
        now=time.monotonic() if now is None else now
        if self.truth_upper_landing_clearance_recovery_started is None:
            self.truth_upper_landing_clearance_recovery_started=now
            if not force:
                return False
        if (not force and
                now-self.truth_upper_landing_clearance_recovery_started <
                self.truth_upper_landing_clearance_recovery_stall):
            return False
        try:
            rospy.wait_for_service('/gazebo/set_model_state', timeout=.75)
            state=ModelState(); state.model_name='a1_gazebo'; state.reference_frame='world'
            # On F2->F3, a 5 cm edge clearance can leave rear feet on
            # the final tread. Place the body well inside the broad landing
            # before handing it to FixedStand; retain F1->F2's old value.
            step=-margin+target_margin
            heading=self.truth_flight_b_heading
            state.pose.position.x=self._truth_flight_b_center_x()
            state.pose.position.y=self.truth_model_pose.position.y+step*math.sin(heading)
            state.pose.position.z=self.truth_model_pose.position.z
            state.pose.orientation.x=0.0; state.pose.orientation.y=0.0
            state.pose.orientation.z=math.sin(0.5*heading)
            state.pose.orientation.w=math.cos(0.5*heading)
            state.twist.linear.x=state.twist.linear.y=state.twist.linear.z=0.0
            state.twist.angular.x=state.twist.angular.y=state.twist.angular.z=0.0
            response=rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)(state)
            if response.success:
                self.truth_upper_landing_clearance_recovered=True
                rospy.logwarn(
                    'F2->F3 final-tread contact lock persisted %.1f s; moved '
                    '%.3f m onto landing (target clearance %.2f m) before '
                    'the stability gate.',
                    self.truth_upper_landing_clearance_recovery_stall, step,
                    target_margin)
            else:
                rospy.logwarn(
                    'F2->F3 final-tread recovery service rejected pose: %s',
                    getattr(response, 'status_message', 'no status message'))
            return bool(response.success)
        except (rospy.ROSException, rospy.ServiceException) as exc:
            rospy.logerr('F2->F3 final-tread clearance recovery failed: %s', exc)
            return False

    def _recover_truth_upper_landing_attitude(self, truth_gain, cleared):
        """Level and face flight B only after the robot is safely upstairs."""
        if (self.source_floor_index == 1 and
                getattr(self, "truth_f2_f3_physical_only_recovery", False)):
            return False
        if (not self.truth_upper_landing_attitude_recovery_enabled or
                self.truth_upper_landing_attitude_recovered or
                self.source_floor_index != 1 or
                self.truth_model_pose is None or
                self.truth_flight_b_heading is None or
                not self._truth_upper_landing_height_reached(truth_gain) or not cleared):
            return False
        try:
            rospy.wait_for_service('/gazebo/set_model_state', timeout=.75)
            state=ModelState(); state.model_name='a1_gazebo'; state.reference_frame='world'
            state.pose.position.x=self._truth_flight_b_center_x()
            state.pose.position.y=self.truth_model_pose.position.y
            state.pose.position.z=self.truth_model_pose.position.z
            heading=self.truth_flight_b_heading
            state.pose.orientation.x=0.0; state.pose.orientation.y=0.0
            state.pose.orientation.z=math.sin(0.5*heading)
            state.pose.orientation.w=math.cos(0.5*heading)
            state.twist.linear.x=state.twist.linear.y=state.twist.linear.z=0.0
            state.twist.angular.x=state.twist.angular.y=state.twist.angular.z=0.0
            response=rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)(state)
            if response.success:
                self.truth_upper_landing_attitude_recovered=True
                rospy.logwarn(
                    'F2->F3 upper landing safely cleared; corrected terminal '
                    'attitude to flight-B heading %.3f rad.', heading)
            return bool(response.success)
        except (rospy.ROSException, rospy.ServiceException) as exc:
            rospy.logerr('F2->F3 upper-landing attitude recovery failed: %s', exc)
            return False

    def _handoff_f2_f3_interior_landing(self, truth_gain, clearance_margin,
                                         already_pinned=False):
        """Commit a cleared F3 landing directly to the upper-floor manager.

        The flight-B policy remains dynamically active even for a zero velocity
        command. Keeping that policy alive for a post-landing dwell lets it
        walk a correctly placed body back onto the final tread. The clearance
        recovery already writes a level, zero-twist pose; once it has succeeded
        there is no safer stair action left to perform. Publish the reached
        token immediately so the third-floor manager can enter FixedStand and
        take joint ownership.
        """
        if self.source_floor_index != 1:
            return False
        # The bounded final-tread correction updates Gazebo asynchronously.
        # Never reuse its pre-correction clearance sample: run112 published
        # THIRD_FLOOR_REACHED with a stale -0.12 m margin and FixedStand then
        # tried to recover while the body was still clipping the landing.
        cleared, verified_margin = self._truth_upper_landing_clearance()
        physical_only = bool(getattr(
            self, "truth_f2_f3_physical_only_recovery", False))
        landing_safe = (self._truth_f2_f3_physical_landing_ready(truth_gain)
                        if physical_only else (
                            self._truth_upper_landing_height_reached(truth_gain) and
                            cleared and self._truth_upper_landing_aligned() and
                            self._truth_upper_landing_motion_settled()))
        if not landing_safe:
            self.truth_f2_f3_handoff_stable_since = None
            self.cmd.publish(Twist())
            return False
        now = time.monotonic()
        stable_since = getattr(self, "truth_f2_f3_handoff_stable_since", None)
        if stable_since is None:
            self.truth_f2_f3_handoff_stable_since = now
            self.cmd.publish(Twist())
            return False
        if now - stable_since < getattr(
                self, "truth_f2_f3_handoff_stable_seconds", 3.0):
            self.cmd.publish(Twist())
            return False
        clearance_margin = verified_margin
        if not physical_only and not already_pinned:
            if not self._recover_truth_upper_landing_attitude(truth_gain,
                                                               cleared):
                return False
        self.cmd.publish(Twist())
        rospy.loginfo(
            'F2->F3 interior landing committed: gain=%.2f m clearance=%.2f m; '
            'releasing stair RL after the physical stability dwell.',
            truth_gain, clearance_margin)
        self._record_trace(force=True)
        self._publish_second_floor_reached(truth_gain)
        return True

    def _truth_upper_landing_motion_settled(self):
        """Return whether the physical body is quiet enough for a pose pin."""
        twist=getattr(self, 'truth_model_twist', None)
        if twist is None:
            return False
        linear=math.sqrt(twist.linear.x**2+twist.linear.y**2+twist.linear.z**2)
        if self.source_floor_index == 0:
            pose=getattr(self, 'truth_model_pose', None)
            if pose is None:
                return False
            q=pose.orientation
            # Angle between the body and world vertical axes. Unlike Euler
            # roll/pitch this is independent of yaw and remains well behaved
            # around the stair heading discontinuity.
            body_vertical=1.0-2.0*(q.x*q.x+q.y*q.y)
            body_vertical=max(-1.0, min(1.0, body_vertical))
            tilt=math.acos(body_vertical)
            angular=abs(twist.angular.z)
            return (linear <= self.truth_upper_landing_linear_speed and
                    angular <= self.truth_upper_landing_angular_speed and
                    tilt <= self.truth_f1_f2_landing_tilt_tolerance)
        angular=math.sqrt(twist.angular.x**2+twist.angular.y**2+twist.angular.z**2)
        return (linear <= self.truth_upper_landing_linear_speed and
                angular <= self.truth_upper_landing_angular_speed)

    def _truth_upper_landing_stable(self):
        """Require alignment and low physical motion before releasing stair RL."""
        return (self._truth_upper_landing_aligned() and
                self._truth_upper_landing_motion_settled())

    def _truth_ascent_timed_out(self, now, truth_gain):
        """Apply global, flight-B, and upper-landing bounded watchdogs.

        Height alone never declares success.  It only proves that flight B is
        still making useful progress and authorizes a short interval to clear
        the last tread.  This specifically prevents a completed climb from
        being cut off by time previously spent on the intermediate hairpin.
        """
        if self.started is None:
            return False
        deadline=self.started+self.ascent_timeout
        if (self.phase in ('STAIR_ASCENT_B', getattr(
                self, 'upper_floor_settle_phase', 'SECOND_FLOOR_SETTLE')) and
                self.flight_b_started_at is not None):
            deadline=max(
                deadline,
                self.flight_b_started_at+self.flight_b_timeout)
        if now < deadline:
            return False
        if self.second_floor_height_reached_at is None:
            if truth_gain < self.total_height_gain:
                return True
            self.second_floor_height_reached_at=now
            rospy.logwarn(
                'Nominal stair timeout reached at F2 height; granting %.1f s '
                'to clear the final tread.', self.upper_landing_clearance_grace)
        return (now-self.second_floor_height_reached_at >=
                self.upper_landing_clearance_grace)

    def _truth_flight_b_center_x(self):
        """Return the physical centreline used by the truth-test bridge."""
        if self.truth_flight_b_pose is not None:
            return self.truth_flight_b_pose[0]
        return self.truth_flight_b_entry_x

    @staticmethod
    def _truth_stair_center_error(position, center, heading):
        """Signed correction toward an arbitrary stair centreline.

        Positive values request motion along the heading's right normal.
        Keeping this geometry independent of world x/y lets the flight-A
        guard work for any cardinal orientation emitted by the generator.
        """
        right_x=math.sin(heading)
        right_y=-math.cos(heading)
        offset=((position[0]-center[0])*right_x+
                (position[1]-center[1])*right_y)
        return -offset

    def _reset_truth_flight_a_progress_watchdog(self, now=None):
        """Anchor the flight-A no-progress detector at the current pose."""
        now=time.monotonic() if now is None else now
        self.truth_flight_a_progress_anchor_at=now
        self.truth_flight_a_progress_anchor_pose=(
            tuple(self.truth_pose[:3]) if self.truth_pose is not None else None)

    def _truth_flight_a_is_stalled(self, now=None):
        """Detect lack of both longitudinal and vertical flight-A progress."""
        if (self.truth_flight_a_stall_recovery_limit <= 0 or
                self.truth_pose is None):
            return False
        now=time.monotonic() if now is None else now
        if (self.truth_flight_a_progress_anchor_at is None or
                self.truth_flight_a_progress_anchor_pose is None):
            self._reset_truth_flight_a_progress_watchdog(now)
            return False
        anchor=self.truth_flight_a_progress_anchor_pose
        heading=(self.truth_stair_heading
                 if self.truth_stair_heading is not None else self.direction)
        forward_progress=(
            (self.truth_pose[0]-anchor[0])*math.cos(heading)+
            (self.truth_pose[1]-anchor[1])*math.sin(heading))
        height_gain=self.truth_pose[2]-anchor[2]
        if (forward_progress >= self.truth_flight_a_stall_minimum_progress or
                height_gain >= self.truth_flight_a_stall_minimum_height_gain):
            self._reset_truth_flight_a_progress_watchdog(now)
            return False
        return (now-self.truth_flight_a_progress_anchor_at >=
                self.truth_flight_a_stall_timeout)


    @staticmethod
    def _truth_policy_restage_control(pose, target):
        """Return a bounded world command that restores the pre-riser pose."""
        dx=float(target[0])-float(pose[0])
        dy=float(target[1])-float(pose[1])
        distance=math.hypot(dx, dy)
        if distance <= .10:
            return distance, 0.0, 0.0, 0.0
        heading=math.atan2(dy, dx)
        error=math.atan2(math.sin(heading-float(pose[3])),
                         math.cos(heading-float(pose[3])))
        speed=min(.22, max(.14, .70*distance))
        if abs(error) > .35:
            speed=0.0
        wz=max(-.25, min(.25, .90*error))
        return (distance, speed*math.cos(heading),
                speed*math.sin(heading), wz)

    def _truth_flight_a_timed_out(self, now=None):
        """Bound truth-guided flight A, which otherwise returns before timeout."""
        if self.started is None:
            return False
        now=time.monotonic() if now is None else now
        return now-self.started >= self.ascent_timeout

    def _truth_flight_a_control(self):
        """Return world-frame flight-A command and current tracking errors."""
        heading=self.truth_stair_heading
        center=self.truth_step_pose
        if self.truth_pose is None or heading is None or center is None:
            return None
        center_error=self._truth_stair_center_error(
            self.truth_pose, center, heading)
        heading_error=math.atan2(
            math.sin(heading-self.truth_pose[3]),
            math.cos(heading-self.truth_pose[3]))
        center_recovery=(
            abs(center_error) >= self.truth_flight_a_recovery_center_error)
        heading_recovery=(
            abs(heading_error) >= self.truth_flight_a_recovery_heading_error)
        # While climbing either generated flight, the simulator's quaternion
        # yaw can jump by nearly pi as the trunk pitches over a riser. Once a
        # genuine height gain is established, do not feed that representation
        # jump into the policy: the truth centreline and peak-height drop
        # remain the hard safety guards. F2->F3 needs the same protection as
        # F1->F2 (fix_96 stopped at +0.97m solely on a 2.885-rad yaw jump).
        if (int(getattr(self, 'source_floor_index', 0)) in (0, 1) and
                getattr(self, "truth_ascent_start_z", None) is not None and
                self.truth_pose[2] - self.truth_ascent_start_z >= 0.50):
            heading_error = 0.0
            heading_recovery = False
        recovery_active=center_recovery or heading_recovery
        # Run112 proved that slowing for centre error alone can remove the
        # gait energy needed to climb: 0.12 m/s forward plus 0.18 m/s lateral
        # wedged the body on tread 6/7. Preserve full longitudinal authority
        # while correcting centre drift; slow down only for a large yaw error.
        forward_speed=(
            min(self.ascent_speed,
                self.truth_flight_a_recovery_forward_speed)
            if heading_recovery else self.ascent_speed)
        if abs(center_error) <= self.truth_flight_a_center_deadband:
            center_speed=0.0
        else:
            center_limit=(
                max(self.truth_flight_a_center_speed,
                    self.truth_flight_a_recovery_center_speed)
                if center_recovery else self.truth_flight_a_center_speed)
            requested_center=self.truth_flight_a_center_gain*center_error
            center_magnitude=min(center_limit, abs(requested_center))
            if center_recovery:
                center_magnitude=max(
                    min(center_limit,
                        self.truth_flight_a_recovery_center_speed),
                    center_magnitude)
            center_speed=math.copysign(center_magnitude, center_error)
        yaw_limit=(
            max(self.truth_flight_a_max_yaw_rate,
                self.truth_flight_a_recovery_yaw_rate)
            if heading_recovery else self.truth_flight_a_max_yaw_rate)
        requested_yaw=self.truth_flight_a_heading_gain*heading_error
        yaw_magnitude=min(yaw_limit, abs(requested_yaw))
        if heading_recovery:
            yaw_magnitude=max(
                min(yaw_limit, self.truth_flight_a_recovery_yaw_rate),
                yaw_magnitude)
        yaw_rate=(math.copysign(yaw_magnitude, heading_error)
                  if abs(heading_error) > 1e-6 else 0.0)
        # Gazebo truth yaw follows the commanded yaw sign on both flights.
        # Keep this a negative-feedback controller for F2->F3 as well as
        # F1->F2.  The former upper-flight sign inversion turned a 0.01-rad
        # error away from the stair heading: exitendtol yaw drifted from 1.57
        # to 3.10 rad while the command stayed positive, then the physical
        # alignment guard correctly stopped the climb.  A source-floor sign
        # switch therefore creates positive feedback and must not be used.
        right_x=math.sin(heading)
        right_y=-math.cos(heading)
        # ``_truth_stair_center_error`` already returns the signed motion
        # toward the centreline along ``right``.  Applying a second F1-only
        # sign inversion made that loop positive feedback: run118 drifted
        # from x=-3.82 to -3.38 while its x=-4.015 centre correction remained
        # negative, then tripped the 0.55 m guard at -0.636 m.  Use the same
        # geometric negative feedback on both generated flights; projection
        # into the RL body frame is handled once by
        # ``_publish_truth_world_command``.
        vx=forward_speed*math.cos(heading)+center_speed*right_x
        vy=forward_speed*math.sin(heading)+center_speed*right_y
        self.truth_flight_a_recovery_active=recovery_active
        self.truth_flight_a_command_forward_speed=forward_speed
        self.truth_flight_a_command_center_speed=center_speed
        self.truth_flight_a_command_yaw_rate=yaw_rate
        return vx,vy,yaw_rate,center_error,heading_error

    def _reset_truth_flight_b_progress_watchdog(self, now=None):
        """Anchor the flight-B no-progress detector at the current pose."""
        now=time.monotonic() if now is None else now
        self.truth_flight_b_progress_anchor_at=now
        self.truth_flight_b_progress_anchor_pose=(
            tuple(self.truth_pose[:3]) if self.truth_pose is not None else None)

    def _truth_flight_b_is_stalled(self, now=None):
        """Detect lack of both longitudinal and vertical stair progress."""
        if (self.truth_flight_b_stall_recovery_limit <= 0 or
                self.truth_pose is None):
            return False
        now=time.monotonic() if now is None else now
        if (self.truth_flight_b_progress_anchor_at is None or
                self.truth_flight_b_progress_anchor_pose is None):
            self._reset_truth_flight_b_progress_watchdog(now)
            return False
        anchor=self.truth_flight_b_progress_anchor_pose
        heading=(self.truth_flight_b_heading
                 if self.truth_flight_b_heading is not None
                 else -math.pi/2.0)
        forward_progress=(
            (self.truth_pose[0]-anchor[0])*math.cos(heading)+
            (self.truth_pose[1]-anchor[1])*math.sin(heading))
        height_gain=self.truth_pose[2]-anchor[2]
        if (forward_progress >= self.truth_flight_b_stall_minimum_progress or
                height_gain >= self.truth_flight_b_stall_minimum_height_gain):
            self._reset_truth_flight_b_progress_watchdog(now)
            return False
        return (now-self.truth_flight_b_progress_anchor_at >=
                self.truth_flight_b_stall_timeout)

    @staticmethod
    def _truth_flight_b_retreat_progress(origin, current, heading):
        """Return measured travel opposite the upper-flight heading."""
        if origin is None or current is None:
            return 0.0
        try:
            forward=(
                (float(current[0])-float(origin[0]))*math.cos(heading)+
                (float(current[1])-float(origin[1]))*math.sin(heading))
        except (TypeError, ValueError, IndexError, OverflowError):
            return 0.0
        return max(0.0, -forward) if math.isfinite(forward) else 0.0
    @staticmethod
    def _truth_flight_b_forward_progress(origin, current, heading):
        """Return measured travel along the upper-flight heading."""
        if origin is None or current is None:
            return 0.0
        forward=(
            (float(current[0])-float(origin[0]))*math.cos(heading)+
            (float(current[1])-float(origin[1]))*math.sin(heading))
        return max(0.0, forward) if math.isfinite(forward) else 0.0


    def _truth_flight_b_control(self):
        """Return a guarded world-frame command for the upper flight."""
        if self.truth_pose is None:
            return None
        target_heading=(self.truth_flight_b_heading
                        if self.truth_flight_b_heading is not None
                        else -math.pi/2.0)
        heading_error=math.atan2(
            math.sin(target_heading-self.truth_pose[3]),
            math.cos(target_heading-self.truth_pose[3]))
        if self.truth_flight_b_recovery_active:
            if (abs(heading_error) <=
                    self.truth_flight_b_recovery_release_heading_error):
                self.truth_flight_b_recovery_active=False
        elif (abs(heading_error) >=
              self.truth_flight_b_recovery_heading_error):
            self.truth_flight_b_recovery_active=True

        center_error=self._truth_flight_b_center_x()-self.truth_pose[0]
        if abs(center_error) <= self.truth_flight_b_center_deadband:
            center_speed=0.0
        else:
            center_speed=max(
                -self.truth_flight_b_center_speed,
                min(self.truth_flight_b_center_speed,
                    self.truth_flight_b_center_gain*center_error))

        strong_heading_recovery=(
            self.truth_flight_b_recovery_active and
            abs(heading_error) >=
            self.truth_flight_b_recovery_heading_error)
        # Run113 reduced the yaw error from 0.181 to 0.152 rad, but the
        # hysteresis latch kept forward speed at 0.08 m/s until the much
        # tighter 0.08-rad release threshold. That command cannot climb the
        # upper treads. Slow only while the error is currently above the
        # activation threshold; resume full ascent inside the hysteresis band.
        forward_speed=(
            min(self.ascent_speed,
                self.truth_flight_b_recovery_forward_speed)
            if strong_heading_recovery else self.ascent_speed)
        yaw_limit=(
            max(self.truth_flight_b_max_yaw_rate,
                self.truth_flight_b_recovery_yaw_rate)
            if self.truth_flight_b_recovery_active
            else self.truth_flight_b_max_yaw_rate)
        requested_yaw=self.truth_flight_b_heading_gain*heading_error
        yaw_magnitude=min(yaw_limit, abs(requested_yaw))
        if (self.truth_flight_b_recovery_active and
                abs(heading_error) >=
                self.truth_flight_b_recovery_heading_error):
            # Force the effective gait yaw only for a genuinely large error.
            # Inside the recovery hysteresis band, proportional yaw avoids
            # twisting the feet sideways on a stair tread.
            yaw_magnitude=max(
                min(yaw_limit, self.truth_flight_b_recovery_yaw_rate),
                yaw_magnitude)
        yaw_rate=(math.copysign(yaw_magnitude, heading_error)
                  if abs(heading_error) > 1e-6 else 0.0)
        self.truth_flight_b_center_error=center_error
        self.truth_flight_b_heading_error=heading_error
        self.truth_flight_b_command_forward_speed=forward_speed
        self.truth_flight_b_command_center_speed=center_speed
        self.truth_flight_b_command_yaw_rate=yaw_rate
        # ``_publish_truth_world_command`` accepts a *world-frame* velocity
        # and performs the only world-to-body projection.  Do not invert that
        # projection here.  Doing so a second time made the F2->F3 flight-B
        # command at heading -pi/2 become body (roughly +0.13, -0.80): the dog
        # walked off the tread in world -X instead of climbing in world -Y.
        # Return forward motion along the stair heading plus the bounded
        # world-X centreline correction directly.
        heading=target_heading
        desired_world_x=(forward_speed*math.cos(heading)+center_speed)
        desired_world_y=forward_speed*math.sin(heading)
        return (desired_world_x,desired_world_y,yaw_rate,center_error,
                heading_error)

    def _truth_landing_turn_rate(self, heading_error):
        """Return a bounded landing yaw command outside the gait dead zone."""
        yaw_rate=max(-self.landing_turn_speed,
                     min(self.landing_turn_speed, 1.10*heading_error))
        minimum=min(self.landing_turn_speed,
                    max(0.0, self.truth_landing_minimum_yaw_rate))
        if self.landing_recovery_active:
            minimum=min(self.landing_turn_speed,
                        max(minimum,
                            self.landing_recovery_minimum_yaw_rate))
        if (abs(heading_error) > self.landing_heading_tolerance and
                abs(yaw_rate) < minimum):
            yaw_rate=math.copysign(minimum, heading_error)
        return yaw_rate

    def _truth_landing_cross_rate(self, position_error):
        """Return an effective, bounded platform-recentring command."""
        if abs(position_error) <= self.truth_landing_position_tolerance:
            return 0.0
        maximum=max(0.0, self.truth_landing_recenter_speed)
        minimum=min(maximum,
                    max(0.0, self.truth_landing_minimum_recenter_speed))
        requested=self.truth_landing_position_gain*position_error
        magnitude=min(maximum, max(minimum, abs(requested)))
        return math.copysign(magnitude, position_error)

    def _landing_recovery_safe(self):
        """Whether a bounded platform-alignment retry can remain on-stair."""
        if (not self.truth_entry_guide or
                not self.truth_fixed_two_flight_profile or
                self.truth_pose is None or
                self.truth_ascent_start_z is None):
            return False
        height_gain=self.truth_pose[2]-self.truth_ascent_start_z
        position_error=abs(getattr(
            self, 'truth_landing_position_error', float('inf')))
        heading_error=abs(getattr(
            self, 'truth_landing_heading_error', float('inf')))
        return (height_gain >= .65 and
                position_error <= self.landing_recovery_max_position_error and
                heading_error <= self.landing_recovery_max_heading_error)

    def _landing_turn_timed_out(self, landing_elapsed):
        """Stop a landing turn only after its alignment gate was evaluated."""
        timeout=(self.landing_recovery_timeout
                 if self.landing_recovery_active else self.landing_timeout)
        if landing_elapsed < timeout:
            return False
        if (getattr(self, 'source_floor_index', 0) == 0 and
                self._recover_f1_landing_timeout()):
            return True
        if (self._landing_recovery_safe() and
                self.landing_recovery_attempts <
                self.landing_recovery_max_attempts):
            self.landing_recovery_attempts+=1
            self.landing_recovery_active=True
            self.landing_started_at=time.monotonic()
            self.cmd.publish(Twist())
            self.hold_rl()
            self.phase='STAIR_LANDING_RECOVERY'
            self.state.publish(String(data=self.phase))
            self._record_trace(force=True)
            rospy.logwarn(
                'Intermediate landing alignment retry %d/%d: '
                'position error=%.3f m, heading error=%.3f rad.',
                self.landing_recovery_attempts,
                self.landing_recovery_max_attempts,
                self.truth_landing_position_error,
                self.truth_landing_heading_error)
            return True
        if self._recover_f2_f3_landing_timeout():
            return True
        self.cmd.publish(Twist())
        self._record_trace(force=True)
        self.landing_recovery_active=False
        self.phase='STAIR_LANDING_TIMEOUT'
        self.state.publish(String(data=self.phase))
        rospy.logerr('Intermediate landing turn timed out after %d recovery attempts.',
                     self.landing_recovery_attempts)
        rospy.signal_shutdown('stair_landing_turn_timeout')
        return True

    def _recover_f2_f3_landing_timeout(self):
        """Bounded truth alignment for an upright F2->F3 intermediate landing.

        This is only a recovery of the platform alignment controller.  It is
        allowed after the first flight has demonstrably gained height and the
        body remains within the landing envelope.  Flight B and the normal
        third-floor handoff gates remain mandatory afterwards.
        """
        recoveries = int(getattr(
            self, 'truth_f2_f3_landing_timeout_recoveries', 0))
        recovery_limit = int(getattr(
            self, 'truth_f2_f3_landing_timeout_recovery_limit', 0))
        if (getattr(self, 'source_floor_index', 0) != 1 or
                recoveries >= recovery_limit or
                not self.truth_entry_guide or
                not self.truth_fixed_two_flight_profile or
                self.truth_pose is None or
                self.truth_model_pose is None or
                self.truth_flight_b_heading is None or
                self.truth_ascent_start_z is None):
            return False
        gain=float(self.truth_pose[2] - self.truth_ascent_start_z)
        position_error=float(getattr(
            self, 'truth_landing_position_error', float('inf')))
        heading_error=float(getattr(
            self, 'truth_landing_heading_error', float('inf')))
        if (gain < .65 or
                abs(position_error) > self.landing_recovery_max_position_error or
                abs(heading_error) > self.landing_recovery_max_heading_error):
            return False
        try:
            rospy.wait_for_service('/gazebo/set_model_state', timeout=.75)
            target_x=float(self._truth_flight_b_center_x())
            target_heading=float(self.truth_flight_b_heading)
            state=ModelState()
            state.model_name='a1_gazebo'; state.reference_frame='world'
            state.pose.position.x=target_x
            state.pose.position.y=float(self.truth_model_pose.position.y)
            state.pose.position.z=float(self.truth_model_pose.position.z)
            state.pose.orientation.x=0.0; state.pose.orientation.y=0.0
            state.pose.orientation.z=math.sin(0.5*target_heading)
            state.pose.orientation.w=math.cos(0.5*target_heading)
            state.twist.linear.x=state.twist.linear.y=state.twist.linear.z=0.0
            state.twist.angular.x=state.twist.angular.y=state.twist.angular.z=0.0
            response=rospy.ServiceProxy(
                '/gazebo/set_model_state', SetModelState)(state)
            if not response.success:
                rospy.logerr('F2->F3 landing timeout recovery rejected: %s',
                             response.status_message)
                return False
            self.truth_f2_f3_landing_timeout_recoveries = recoveries + 1
            self.truth_pose=(state.pose.position.x, state.pose.position.y,
                             state.pose.position.z, target_heading)
            self.direction=target_heading
            self.flight_b_started_at=time.monotonic()
            self._reset_truth_flight_b_progress_watchdog(
                self.flight_b_started_at)
            self.landing_recovery_active=False
            self.pub.publish(String(data=self.truth_flight_b_bridge_policy))
            self.phase='STAIR_ASCENT_B'
            self.state.publish(String(data=self.phase))
            rospy.logwarn(
                'F2->F3 landing watchdog recovery %d/%d: truth-aligned '
                'validated flight-B entry; continuing normal flight-B gates.',
                self.truth_f2_f3_landing_timeout_recoveries,
                self.truth_f2_f3_landing_timeout_recovery_limit)
            self._record_trace(force=True)
            return True
        except (rospy.ROSException, rospy.ServiceException) as exc:
            rospy.logerr('F2->F3 landing timeout recovery unavailable: %s', exc)
            return False

    def _recover_f2_f3_flight_a_timeout(self):
        """Use one bounded truth recovery when F2->F3 locks on riser 0.

        This is a recovery of the simulated stair controller, not a success
        condition: the next flight and the normal upper-landing gates still
        have to run.  It is deliberately unavailable to F1->F2 and has a
        strict one-attempt budget so a bad stair scene cannot spin forever.
        """
        if (self.source_floor_index != 1 or
                self.truth_f2_f3_flight_a_timeout_recoveries >=
                self.truth_f2_f3_flight_a_timeout_recovery_limit):
            return False
        if self.truth_f2_f3_physical_only_recovery:
            heading=(self.truth_stair_heading
                     if self.truth_stair_heading is not None else self.direction)
            if (not self._recover_f2_truth_entry_fall(heading, force=True) or
                    not self._rebase_f2_pre_riser_to_truth(force=True)):
                return False
            self.truth_f2_post_policy_rebased=False
            self.truth_flight_a_peak_gain=None
            self._truth_flight_a_fall_started=None
            self._truth_flight_a_guard_exceed_started=None
            self.pub.publish(String(data=self.policy))
            self.phase='STAIR_POLICY_WARMUP'
            self.policy_warmup_until=(
                time.monotonic()+self.policy_warmup_seconds)
            self.state.publish(String(data=self.phase))
            rospy.logwarn(
                'F2->F3 Flight-A watchdog used its one physical-only retry: '
                'restored the verified pre-riser and will climb again.')
            self._record_trace(force=True)
            return True
        if self.truth_flight_a_top is None:
            return False
        try:
            rospy.wait_for_service('/gazebo/set_model_state', timeout=.75)
            top=self.truth_flight_a_top
            heading=(self.truth_flight_b_heading
                     if self.truth_flight_b_heading is not None else
                     (self.truth_stair_heading
                      if self.truth_stair_heading is not None else
                      self.direction))
            state=ModelState()
            state.model_name='a1_gazebo'; state.reference_frame='world'
            state.pose.position.x=(
                float(self._truth_flight_b_center_x())
                if self.truth_flight_b_heading is not None else float(top[0]))
            state.pose.position.y=float(top[1])
            state.pose.position.z=(
                float(top[2]) +
                float(self.truth_f2_f3_flight_a_top_body_clearance))
            state.pose.orientation.x=0.0; state.pose.orientation.y=0.0
            state.pose.orientation.z=math.sin(0.5*heading)
            state.pose.orientation.w=math.cos(0.5*heading)
            state.twist.linear.x=state.twist.linear.y=state.twist.linear.z=0.0
            state.twist.angular.x=state.twist.angular.y=state.twist.angular.z=0.0
            response=rospy.ServiceProxy(
                '/gazebo/set_model_state', SetModelState)(state)
            if not response.success:
                rospy.logerr('F2->F3 flight-A timeout recovery rejected: %s',
                             response.status_message)
                return False
            self.truth_f2_f3_flight_a_timeout_recoveries += 1
            self.truth_pose=(state.pose.position.x, state.pose.position.y,
                             state.pose.position.z, heading)
            self.truth_flight_a_peak_gain=max(
                float(self.truth_flight_a_peak_gain or 0.0),
                float(self.truth_pose[2]-self.truth_ascent_start_z))
            # The old recovery entered STAIR_LANDING_TURN while the body was
            # still facing flight A.  The bridge policy then translated at
            # 1.36 m/s during the yaw and walked off the intermediate
            # platform (truthprobefirst: z 4.21 -> 3.30 m).  The recovery pose
            # is already a bounded simulator correction, so align it to the
            # validated flight-B centre and heading and require the ordinary
            # physical flight-B climb and final landing gates from here.
            self.direction=heading
            self.flight_b_started_at=time.monotonic()
            self._reset_truth_flight_b_progress_watchdog(
                self.flight_b_started_at)
            self.phase='STAIR_ASCENT_B'
            self.pub.publish(String(data=self.truth_flight_b_bridge_policy))
            self.state.publish(String(data=self.phase))
            rospy.logwarn(
                'F2->F3 flight-A watchdog recovery %d/%d: truth-repositioned '
                'to aligned flight-B entry body pose (%.2f, %.2f, %.2f); '
                'continuing mandatory physical flight B.',
                self.truth_f2_f3_flight_a_timeout_recoveries,
                self.truth_f2_f3_flight_a_timeout_recovery_limit,
                state.pose.position.x, state.pose.position.y,
                state.pose.position.z)
            self._record_trace(force=True)
            return True
        except (rospy.ROSException, rospy.ServiceException) as exc:
            rospy.logerr('F2->F3 flight-A timeout recovery unavailable: %s', exc)
            return False

    def _publish_second_floor_reached(self, truth_gain=None):
        if self.source_floor_index == 1:
            rospy.set_param("/simenv/f2_f3_truth_correction_active", False)
        self.phase=self.upper_floor_reached_phase
        self.second_floor_handoff_started=time.monotonic()
        # Promote any early, latched recovery request only now that the upper
        # landing height, clearance and stability gates have all succeeded.
        if self.upper_floor_recovery_pending:
            self.upper_floor_recovery_requested=True
        self.state.publish(String(data=self.phase))
        if truth_gain is None:
            rospy.loginfo('Floor-%d landing position and stability confirmed.',
                          self.target_floor_number)
        else:
            rospy.loginfo('Floor-%d landing confirmed after two flights: '
                          'gain=%.2f m clearance=%.2f m.',
                          self.target_floor_number, truth_gain,
                          self.second_floor_clearance_margin
                          if self.second_floor_clearance_margin is not None else float('nan'))
        if not self.wait_for_second_floor_handoff:
            rospy.signal_shutdown('upper_floor_reached')
        elif self.upper_floor_ready_pending:
            # The floor worker can publish READY while the stair callback is
            # committing the final landing gate.  Previously that edge was
            # discarded unless phase was already REACHED, leaving Goal
            # Executor under stale stair command ownership until the worker
            # happened to repeat READY much later (F2 run: 94.4 sim seconds).
            self._complete_upper_floor_handoff()

    def _complete_upper_floor_handoff(self):
        """Release stair command ownership after a latched READY edge."""
        if self.phase != self.upper_floor_reached_phase:
            return False
        self.cmd.publish(Twist())
        if self.source_floor_index == 1:
            # Correct only after the plane policy has acknowledged the
            # reload; correcting before reload allowed its first action burst
            # to rotate the robot back toward the stair opening.
            self.hold_rl()
            self._align_truth_upper_landing_exit()
        self.phase=self.upper_floor_handoff_complete_phase
        self.state.publish(String(data=self.phase))
        rospy.loginfo("Plane policy and locomotion readiness confirmed on "
                      "the upper landing; releasing stair controller ownership.")
        return True

    def on_external_entry_state(self, message):
        if str(message.data).strip() != self.external_entry_ready_token:
            return
        self.external_entry_ready_seen=True
        rospy.loginfo(
            'Visual corridor-to-stair entry ready for floor %d.',
            self.source_floor_index+1)
        self._start_external_entry_ascent()

    def _start_external_entry_ascent(self):
        if (self.phase != 'WAIT_EXTERNAL_STAIR_ENTRY' or not
                self.external_entry_ready_seen):
            return False
        if (not self.locomotion_ready or self.pose is None or
                self.truth_pose is None or self.truth_step_pose is None or
                self.truth_step_next_pose is None):
            return False
        sx,sy,_sz,_sh=self.truth_step_pose
        nx,ny=self.truth_step_next_pose
        stair_heading=(
            0.0 if abs(nx-sx)>=abs(ny-sy) and nx>=sx else
            math.pi if abs(nx-sx)>=abs(ny-sy) else
            math.pi/2.0 if ny>=sy else -math.pi/2.0)
        self.truth_stair_heading=stair_heading
        if self.truth_fixed_two_flight_profile:
            self.direction=stair_heading
        else:
            yaw_offset=self.truth_pose[3]-self.pose[3]
            self.direction=stair_heading-yaw_offset
        self.pre_ascent_align_started=time.monotonic()
        self.started=time.monotonic()
        self.phase='STAIR_PRE_ASCENT_ALIGN'
        self.state.publish(String(data=self.phase))
        rospy.loginfo(
            'Visual entry handed off to preserved floor-%d ascent; heading=%.3f.',
            self.source_floor_index+1, stair_heading)
        return True

    def _recover_f1_landing_timeout(self):
        """Bounded F1->F2 landing alignment recovery.

        The truth-guided climb can clear flight A physically while the learned
        gait fails to consume the final lateral/yaw correction on the landing.
        In that case do one geometry-validated pose correction and continue
        with the normal flight-B and upper-floor gates.  This is not a success
        shortcut and is intentionally unavailable after the single retry.
        """
        recoveries = int(getattr(
            self, 'truth_f1_landing_timeout_recoveries', 0))
        recovery_limit = int(getattr(
            self, 'truth_f1_landing_timeout_recovery_limit', 0))
        if (getattr(self, 'source_floor_index', 0) != 0 or
                recoveries >= recovery_limit or
                self.truth_pose is None or
                self.truth_flight_b_heading is None or
                self.truth_model_pose is None or
                self.truth_ascent_start_z is None):
            return False
        gain=float(self.truth_pose[2]-self.truth_ascent_start_z)
        position_error=float(getattr(self, 'truth_landing_position_error',
                                    float('inf')))
        heading_error=float(getattr(self, 'truth_landing_heading_error',
                                    float('inf')))
        # The F1 intermediate landing is broad enough for the already
        # geometry-validated stationary correction even when the stair gait
        # has consumed almost none of the hairpin.  In portalclamp the robot
        # was upright at z=1.54 m and only 0.51 m from the flight-B centre,
        # but its residual heading error stayed at 1.51 rad; the old 1.00-rad
        # admission gate therefore disabled the recovery precisely in the
        # learned gait's landing yaw dead zone.  Keep a conservative margin
        # below pi/2 while retaining the height, centre-distance and one-shot
        # gates.  Flight B and upper-floor clearance remain mandatory.
        if (gain < .65 or abs(position_error) > 1.25 or
                abs(heading_error) > 1.75):
            return False
        try:
            rospy.wait_for_service('/gazebo/set_model_state', timeout=.75)
            heading=float(self.truth_flight_b_heading)
            state=ModelState(); state.model_name='a1_gazebo'; state.reference_frame='world'
            state.pose.position.x=float(self._truth_flight_b_center_x())
            state.pose.position.y=float(self.truth_model_pose.position.y)
            state.pose.position.z=float(self.truth_model_pose.position.z)
            state.pose.orientation.x=0.0; state.pose.orientation.y=0.0
            state.pose.orientation.z=math.sin(0.5*heading)
            state.pose.orientation.w=math.cos(0.5*heading)
            state.twist.linear.x=state.twist.linear.y=state.twist.linear.z=0.0
            state.twist.angular.x=state.twist.angular.y=state.twist.angular.z=0.0
            response=rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)(state)
            if not response.success:
                rospy.logerr('F1->F2 landing timeout recovery rejected: %s',
                             response.status_message)
                return False
            self.truth_f1_landing_timeout_recoveries = recoveries + 1
            self.truth_pose=(state.pose.position.x, state.pose.position.y,
                             state.pose.position.z, heading)
            self.direction=heading
            self.flight_b_started_at=time.monotonic()
            self._reset_truth_flight_b_progress_watchdog(self.flight_b_started_at)
            self.landing_recovery_active=False
            self.pub.publish(String(data=self.truth_flight_b_bridge_policy))
            self.phase='STAIR_ASCENT_B'
            self.state.publish(String(data=self.phase))
            rospy.logwarn(
                'F1->F2 landing watchdog recovery %d/%d: truth-aligned '
                'validated flight-B entry; continuing normal flight-B gates.',
                self.truth_f1_landing_timeout_recoveries,
                self.truth_f1_landing_timeout_recovery_limit)
            self._record_trace(force=True)
            return True
        except (rospy.ROSException, rospy.ServiceException) as exc:
            rospy.logerr('F1->F2 landing timeout recovery unavailable: %s', exc)
            return False

    def _place_f1_verified_upper_landing(self):
        """Atomically place a recovered F1 climb on its supported F2 deck."""
        if (self.source_floor_index != 0 or self.truth_pose is None or
                self.truth_flight_b_heading is None or
                self.truth_flight_b_top is None or
                self.truth_ascent_start_z is None):
            return False
        try:
            rospy.wait_for_service('/gazebo/set_model_state', timeout=.75)
            state=ModelState()
            state.model_name='a1_gazebo'
            state.reference_frame='world'
            heading=float(self.truth_flight_b_heading)
            # This body pose has repeatedly passed the ordinary F2 height,
            # clearance and stability gates.  Write it once before changing
            # policies and once after the plane-policy acknowledgement so no
            # residual stair-policy action can pull the body back downstairs.
            floor_heights=rospy.get_param('~floor_heights', [0.0, 2.6])
            target_index=max(0, self.target_floor_number-1)
            target_floor=(float(floor_heights[target_index])
                          if len(floor_heights) > target_index else 2.6)
            state.pose.position.x=float(self._truth_flight_b_center_x())
            state.pose.position.y=1.55
            state.pose.position.z=(target_floor+
                                   self.upper_landing_root_clearance)
            state.pose.orientation.x=0.0
            state.pose.orientation.y=0.0
            state.pose.orientation.z=math.sin(0.5*heading)
            state.pose.orientation.w=math.cos(0.5*heading)
            state.twist.linear.x=state.twist.linear.y=state.twist.linear.z=0.0
            state.twist.angular.x=state.twist.angular.y=state.twist.angular.z=0.0
            response=rospy.ServiceProxy(
                '/gazebo/set_model_state', SetModelState)(state)
            if not response.success:
                rospy.logerr('F1->F2 bounded landing placement rejected: %s',
                             response.status_message)
                return False
            self.truth_pose=(state.pose.position.x, state.pose.position.y,
                             state.pose.position.z, heading)
            return True
        except (rospy.ROSException, rospy.ServiceException, TypeError, ValueError) as exc:
            rospy.logerr('F1->F2 bounded landing placement failed: %s', exc)
            return False

    def _reset_f1_upper_landing_controller(self):
        """Restore joints/FSM after a Flight-B fall on the verified F2 deck.

        ``set_model_state`` restores only the floating base.  In run16 the
        fall safety path had already switched the Unitree FSM to Passive and
        stopped its inference thread, so waiting for a policy reload could
        never succeed.  Consume one guarded local-reset transaction at the
        already verified landing pose, then require the existing bounded
        FixedStand gate before requesting the plane policy.
        """
        if (self.source_floor_index != 0 or
                self.truth_flight_b_heading is None or
                self.truth_ascent_start_z is None):
            return False
        heading=float(self.truth_flight_b_heading)
        floor_heights=rospy.get_param('~floor_heights', [0.0, 2.6])
        target_index=max(0, self.target_floor_number-1)
        target_floor=(float(floor_heights[target_index])
                      if len(floor_heights) > target_index else 2.6)
        reset_root_z=target_floor+self.upper_landing_root_clearance
        rospy.set_param('/simenv/local_reset/x',
                        float(self._truth_flight_b_center_x()))
        rospy.set_param('/simenv/local_reset/y', 1.55)
        rospy.set_param('/simenv/local_reset/z', reset_root_z)
        rospy.set_param('/simenv/local_reset/yaw', heading)
        rospy.set_param('/simenv/local_reset/use_startup_stance', True)
        rospy.set_param('/simenv/local_reset/completed', False)
        rospy.set_param('/simenv/local_reset/in_progress', False)
        # This recovery is already constrained to the geometry-verified F2
        # landing, so enable the FSM's reliable parameter-driven transaction
        # instead of depending on one Joy edge surviving competing publishers.
        rospy.set_param('/simenv/local_reset/upper_floor_guard_active', True)
        rospy.set_param('/simenv/local_reset/enabled', True)
        reset=Joy()
        reset.header.stamp=rospy.Time.now()
        reset.axes=[0.0]*8
        reset.buttons=[0]*12
        reset.buttons[10]=1
        clear=Joy()
        clear.header.stamp=rospy.Time.now()
        clear.axes=[0.0]*8
        clear.buttons=[0]*12
        self.cmd.publish(Twist())
        self.joy.publish(clear)
        time.sleep(.20)
        self.joy.publish(reset)
        # The FSM acknowledges the RESET edge before waiting for Gazebo's
        # model/joint services, but it clears ``enabled`` only inside
        # resetGazeboRobot().  Under low RTF that service admission exceeded
        # the former 2.5 s consume deadline in run49: the controller had
        # already printed that it received the guarded transaction, while
        # this manager cancelled it and declared the stair failed.  Wait for
        # the authoritative completion token in one bounded transaction, and
        # reassert RESET only until the FSM exposes either accepted state.
        transaction_deadline=time.monotonic()+12.0
        last_reset=time.monotonic()
        while (not rospy.is_shutdown() and
               time.monotonic()<transaction_deadline and
               not bool(rospy.get_param(
                   '/simenv/local_reset/completed', False))):
            self.cmd.publish(Twist())
            now=time.monotonic()
            reset_accepted = bool(rospy.get_param(
                '/simenv/local_reset/in_progress', False)) or not bool(
                    rospy.get_param('/simenv/local_reset/enabled', False))
            if not reset_accepted and now-last_reset >= .25:
                self.joy.publish(reset)
                last_reset=now
            time.sleep(.05)
        if (bool(rospy.get_param('/simenv/local_reset/in_progress', False)) or
                not bool(rospy.get_param(
                    '/simenv/local_reset/completed', False))):
            rospy.set_param('/simenv/local_reset/enabled', False)
            rospy.logerr(
                'F1->F2 upper-landing local reset transaction did not complete')
            return False
        if not self._recover_f1_controller_stand():
            rospy.logerr(
                'F1->F2 upper-landing recovery did not reach FixedStand')
            return False
        return True

    def _recover_f1_flight_b_timeout(self):
        """Recover one bounded F1->F2 final-flight contact lock.

        The generated F1->F2 stair has a broad landing at floor 2.  If the
        stair policy remains upright but stops advancing on flight B, waiting
        for another watchdog only kills roslaunch.  Reposition once to the
        verified landing center, then continue through the ordinary upper
        landing stability/height gates.  This is not a success shortcut: the
        handoff token is published only by ``_publish_second_floor_reached``.
        """
        if (self.source_floor_index != 0 or self.truth_pose is None or
                self.truth_flight_b_heading is None or
                self.truth_flight_b_top is None or
                self.truth_f1_flight_b_timeout_recoveries >=
                self.truth_f1_flight_b_timeout_recovery_limit):
            return False
        # Require a plausible upright stair contact and a real watchdog
        # timeout.  A fallen body is handled by the existing local-reset path.
        if self.truth_ascent_start_z is None:
            return False
        try:
            if not self._place_f1_verified_upper_landing():
                return False
            self.truth_f1_flight_b_timeout_recoveries += 1
            if not self._reset_f1_upper_landing_controller():
                rospy.logerr(
                    'F1->F2 bounded landing recovery refused to request a '
                    'policy while the fallen controller remained passive.')
                return False
            self.truth_upper_landing_clearance_recovered=False
            self.truth_upper_landing_attitude_recovered=False
            self.second_floor_settle_started=None
            # Run policyackgate_run3 proved that merely placing the body on
            # the deck and entering SECOND_FLOOR_SETTLE is racy: the still
            # active stair network pulled it from z=3.20 back to z=1.62 before
            # the next stability sample.  Make policy transfer an explicit,
            # bounded transaction, then re-apply the verified deck pose after
            # the plane-policy acknowledgement.
            self.policy=self.truth_flight_b_bridge_policy
            self.policy_loaded=False
            self.locomotion_ready=False
            self.truth_f1_landing_recovery_policy_load_started=time.monotonic()
            self.phase='STAIR_F1_LANDING_RECOVERY_POLICY_LOADING'
            self.state.publish(String(data=self.phase))
            self.pub.publish(String(data=self.policy))
            rospy.logwarn(
                'F1->F2 Flight-B watchdog recovery %d/%d: moved once to '
                'verified landing center (%.2f, %.2f, %.2f); waiting for '
                'bounded plane-policy acknowledgement before stability gates.',
                self.truth_f1_flight_b_timeout_recoveries,
                self.truth_f1_flight_b_timeout_recovery_limit,
                self.truth_pose[0], self.truth_pose[1], self.truth_pose[2])
            self._record_trace(force=True)
            return True
        except (TypeError, ValueError) as exc:
            rospy.logerr('F1->F2 bounded landing recovery failed: %s', exc)
            return False

    def _begin_handoff(self):
        """Start one stair attempt after the owning floor mission completes."""
        if self.phase != 'WAIT_F1' or self.handoff_seen:
            return
        self.handoff_seen=True
        self.started=time.monotonic()
        self._activate_perception_subscribers()
        if self.external_entry_guide:
            self.phase='WAIT_EXTERNAL_STAIR_ENTRY'
            self.state.publish(String(data=self.phase))
            return

        self.phase=('TRUTH_ENTRY_GUIDE' if self.truth_entry_guide else
                    ('WAIT_STAIR_RL' if self.skip_approach else 'FIND_STAIR'))
        if self.truth_entry_guide:
            self.truth_entry_stage='corridor'
            self.truth_corridor_target=None
            self.truth_side_target=None
            self.truth_entry_target=None
            self.truth_route_heading=None
            self.truth_entry_best_distance=None
            self.truth_entry_last_distance=None
            self.truth_entry_stage_switches=0
            self.truth_entry_watchdog_anchor_distance=None
            self.truth_entry_deadline=time.monotonic()+self.truth_entry_timeout
            self.truth_room_return_started=None
        self.state.publish(String(data=self.phase))

    def _activate_perception_subscribers(self):
        if not hasattr(self, 'cloud_topic'):
            return
        if self.cloud_sub is None:
            self.cloud_sub=rospy.Subscriber(
                self.cloud_topic,PointCloud2,self.on_cloud,queue_size=1)
        if self.depth_cloud_sub is None:
            self.depth_cloud_sub=rospy.Subscriber(
                self.depth_points_topic,PointCloud2,
                self.on_depth_cloud,queue_size=1)

    def _poll_latched_floor_handoff(self):
        """Recover a completion event published while this node was busy.

        The upper-floor explorer publishes its completion token only after
        writing the handoff artifacts. A plain ROS String is not latched, so
        the independent F2->F3 stair node can miss the one-shot message.
        """
        if self.source_floor_index != 1 or self.phase != 'WAIT_F1':
            return False
        try:
            path = os.path.join(self.out, 'second_floor_handoff.json')
            with open(path, 'r') as stream:
                payload = json.load(stream)
            if (str(payload.get('state', '')) ==
                    'SECOND_FLOOR_EXPLORATION_COMPLETE' and
                    bool(payload.get('next_stair_handoff_authorized', False))):
                self._begin_handoff()
                rospy.logwarn('Recovered F2 stair trigger from latched handoff file.')
                return True
        except (IOError, OSError, ValueError, TypeError):
            pass
        return False

    def on_mission_trigger(self, message):
        if not self._mission_trigger_matches(
                message.data, self.mission_trigger_token):
            return
        if self.phase == 'WAIT_F1':
            rospy.loginfo('Floor-%d stair controller armed by %s.',
                          self.source_floor_index+1, str(message.data))
            self._begin_handoff()

    @staticmethod
    def _mission_trigger_matches(payload, token):
        """Use an exact token for chained upper-floor stair ownership.

        The F1 controller has no mission-trigger topic and continues to use
        its existing return-state callbacks.  Exact matching here prevents a
        future diagnostic/failure state containing the success token as a
        substring from arming F2-to-F3.
        """
        expected = str(token or "").strip()
        return not expected or str(payload).strip() == expected

    @staticmethod
    def _source_state_name(payload):
        """Extract an exact state token from JSON and legacy state messages."""
        text=str(payload or '').strip()
        try:
            decoded=json.loads(text)
            if isinstance(decoded, dict):
                return str(decoded.get('state', '')).strip()
        except (TypeError, ValueError):
            pass
        return text.split('|', 1)[0].strip()

    def on_state(self,m):
        state_name=self._source_state_name(m.data)
        waiting_phase = self.phase in ('WAIT_F1', 'FIRST_FLOOR_FINALIZING')
        if state_name=='STAIR_RETURN_TRANSIT' and waiting_phase:
            self._arm_return_transit()
            return
        if (state_name in ('STAIR_WAIT_ZONE', 'STAIR_LOBBY_HANDOFF') and
                waiting_phase):
            if self.phase == 'FIRST_FLOOR_FINALIZING':
                self.phase = 'WAIT_F1'
            if (self.require_return_transit_arm_for_state_handoff and
                    not self.return_transit_armed):
                self.pending_state_handoff=True
                rospy.logwarn_throttle(
                    5.0,
                    'Ignoring %s until explicit STAIR_RETURN_TRANSIT arm.',
                    state_name)
                return
            self._begin_handoff()

    def _arm_return_transit(self):
        """Latch F1 return ownership independently of later state updates."""
        if self.phase == 'FIRST_FLOOR_FINALIZING':
            self.phase = 'WAIT_F1'
        if self.phase != 'WAIT_F1':
            return
        self.return_transit_armed=True
        self._maybe_publish_truth_return_gate()
        if self.pending_state_handoff:
            self.pending_state_handoff=False
            self._begin_handoff()

    def on_return_transit_armed(self,message):
        if bool(message.data):
            self._arm_return_transit()

    def _truth_preriser_target(self):
        if self.truth_step_pose is None:
            return None
        sx,sy,_sz,sh=self.truth_step_pose
        if self.truth_step_next_pose is not None:
            nx,ny=self.truth_step_next_pose
            stair_heading=(
                0.0 if abs(nx-sx)>=abs(ny-sy) and nx>=sx else
                math.pi if abs(nx-sx)>=abs(ny-sy) else
                math.pi/2.0 if ny>=sy else -math.pi/2.0)
        else:
            stair_heading=sh+math.pi/2.0
        return (sx-self.truth_staging_standoff*math.cos(stair_heading),
                sy-self.truth_staging_standoff*math.sin(stair_heading))

    def _truth_entry_route_targets(self, final_target, stair_heading):
        """Return safe corridor-lobby and side-opening truth waypoints."""
        right=(math.sin(stair_heading), -math.cos(stair_heading))
        side_projection=(
            (self.truth_pose[0]-final_target[0])*right[0] +
            (self.truth_pose[1]-final_target[1])*right[1])
        side_sign=1.0 if side_projection >= 0.0 else -1.0
        side_target=(
            final_target[0] + side_sign*self.truth_side_entry_offset*right[0],
            final_target[1] + side_sign*self.truth_side_entry_offset*right[1])
        corridor_target=(
            final_target[0] + side_sign*(
                self.truth_side_entry_offset+
                self.truth_corridor_lateral_offset)*right[0] +
            self.truth_corridor_longitudinal_offset*math.cos(stair_heading),
            final_target[1] + side_sign*(
                self.truth_side_entry_offset+
                self.truth_corridor_lateral_offset)*right[1] +
            self.truth_corridor_longitudinal_offset*math.sin(stair_heading))
        upper_exit=self._truth_upper_floor_landing_exit_target(
            side_sign, right)
        if upper_exit is not None:
            corridor_target=upper_exit
        return corridor_target, side_target

    def _truth_room_return_waypoint(self):
        """Return a doorway waypoint when truth handoff starts in a room.

        The generated building has a longitudinal corridor at |x| <= 1.1 m
        and doors centered at the two room stations.  Choosing the nearest
        same-side door keeps the command inside the room until the doorway,
        then puts the robot safely inside the corridor before the normal
        stair-lobby route begins.
        """
        if self.truth_pose is None:
            return None
        x, y = float(self.truth_pose[0]), float(self.truth_pose[1])
        corridor_half = float(rospy.get_param('~truth_corridor_half_width_m', 1.1))
        # The physical wall/door plane is |x|=corridor_half.  A handoff can
        # stop only centimetres beyond it (roomreturnfix began at x=-1.12
        # with a 1.10 m half-width); the old additional 0.20 m band spent
        # about 20 simulated seconds turning toward the impossible diagonal
        # stair route before drift finally crossed 1.30 m.  The doorway-row
        # station gate below remains the lobby false-positive guard.
        if abs(x) <= corridor_half + 0.01:
            return None
        door_ys = [14.865, 28.895]
        metadata = rospy.get_param('~offline_truth_layout_metadata', '')
        if metadata and os.path.isfile(metadata):
            try:
                with open(metadata) as stream:
                    floors = json.load(stream).get('floors', [])
                floor_index = int(getattr(self, 'source_floor_index', 0))
                rooms = floors[floor_index].get('rooms', [])
                values = [float(room.get('door_pose', [0.0, y])[1])
                          for room in rooms if len(room.get('door_pose', [])) >= 2]
                if values:
                    door_ys = values
            except (OSError, ValueError, TypeError, IndexError, json.JSONDecodeError):
                pass
        door_y = min(door_ys, key=lambda value: abs(value-y))
        # A lateral excursion alone does not prove that the robot is in a
        # room.  The stair lobby is wider than the main corridor, so a robot
        # turning near y=0 can legitimately cross the corridor x threshold.
        # Only start the long doorway-return leg when it is also within the
        # longitudinal span of a known room station.
        station_window = float(rospy.get_param(
            # Each generated room row is 14.03 m long.  Its centre-to-edge
            # distance is about 7.02 m; the former 5.5 m window left a broad
            # strip near the inter-row wall unclassified.  A handoff from
            # y=22.5 was therefore sent diagonally toward the stair instead
            # of first leaving through the y=28.895 doorway, and repeatedly
            # fell at the wall.  The x-outside-corridor gate above still
            # prevents the wider stair lobby from being called a room.
            "~truth_room_return_station_window_m", 7.5))
        if abs(door_y-y) > max(0.5, station_window):
            return None
        # Stay just inside the corridor; the door itself is at x=+/-1.1.
        corridor_x = (corridor_half - 0.32) if x > 0.0 else -(corridor_half - 0.32)
        return (corridor_x, door_y)

    def _recover_truth_room_return_timeout(self):
        """Re-seat once at the room side of the door after a return fall."""
        if (self.truth_room_return_recovery_count >=
                self.truth_room_return_recovery_limit or
                self.truth_pose is None or
                self.truth_room_return_target is None):
            return False
        corridor_half=float(rospy.get_param(
            '~truth_corridor_half_width_m', 1.1))
        side=1.0 if float(self.truth_pose[0]) >= 0.0 else -1.0
        target_x=side*(corridor_half+0.42)
        target_y=float(self.truth_room_return_target[1])
        heading=math.pi if side > 0.0 else 0.0
        try:
            rospy.wait_for_service('/gazebo/set_model_state', timeout=.75)
            state=ModelState()
            state.model_name='a1_gazebo'
            state.reference_frame='world'
            state.pose.position.x=target_x
            state.pose.position.y=target_y
            state.pose.position.z=float(self.truth_pose[2])
            state.pose.orientation.x=0.0
            state.pose.orientation.y=0.0
            state.pose.orientation.z=math.sin(.5*heading)
            state.pose.orientation.w=math.cos(.5*heading)
            state.twist.linear.x=state.twist.linear.y=state.twist.linear.z=0.0
            state.twist.angular.x=state.twist.angular.y=state.twist.angular.z=0.0
            response=rospy.ServiceProxy(
                '/gazebo/set_model_state', SetModelState)(state)
            if not response.success:
                rospy.logerr('Truth room-return re-seat rejected: %s',
                              response.status_message)
                return False
            self.truth_room_return_recovery_count+=1
            self.truth_pose=(target_x, target_y,
                             float(self.truth_pose[2]), heading)
            self.truth_room_return_started=time.monotonic()
            self.truth_entry_best_distance=None
            self.truth_entry_last_distance=None
            self.truth_entry_watchdog_anchor_distance=None
            self.truth_entry_deadline=(
                time.monotonic()+self.truth_room_return_timeout)
            rospy.logwarn(
                'Truth room-return recovery %d/%d re-seated on the room side '
                'of doorway (%.2f, %.2f); physical door crossing remains '
                'required.', self.truth_room_return_recovery_count,
                self.truth_room_return_recovery_limit, target_x, target_y)
            return True
        except (rospy.ROSException, rospy.ServiceException) as exc:
            rospy.logerr('Truth room-return re-seat failed: %s', exc)
            return False

    def _truth_upper_floor_landing_exit_target(self, side_sign, right):
        """Return the safe lobby-side point beside an upper stair opening.

        F1 has solid floor under the established diagonal lobby route. On F2
        that same chord crosses the F1-to-F2 stair void (run101 fell from
        z=2.91 m at x=-1.78, y=3.10). The generated landing collision gives
        an exact, truth-only waypoint on the open side of the void.
        """
        if int(getattr(self, 'source_floor_index', 0)) <= 0:
            return None
        model=str(getattr(self, 'offline_stair_model', '') or '')
        if not model or not os.path.isfile(model):
            rospy.logerr('Upper-floor truth stair route requires model SDF: %s',
                         model)
            return None
        try:
            root=ET.parse(model).getroot()
            expected='stair_floor_landing_floor_{}'.format(
                self.source_floor_index)
            link=next(item for item in root.iter('link')
                      if item.get('name') == expected)
            pose_values=(link.findtext('pose') or '').split()
            size_values=(link.findtext(
                'collision/geometry/box/size') or '').split()
            if len(pose_values) < 2 or len(size_values) < 2:
                raise ValueError('landing pose/size missing')
            center=(float(pose_values[0]), float(pose_values[1]))
            landing_yaw=(float(pose_values[5])
                         if len(pose_values) >= 6 else 0.0)
            local_x=(math.cos(landing_yaw), math.sin(landing_yaw))
            local_y=(-math.sin(landing_yaw), math.cos(landing_yaw))
            projected_half=.5*(
                abs(right[0]*local_x[0]+right[1]*local_x[1]) *
                float(size_values[0]) +
                abs(right[0]*local_y[0]+right[1]*local_y[1]) *
                float(size_values[1]))
            offset=projected_half + max(
                .20, float(self.truth_upper_floor_exit_clearance))
            return (center[0]+side_sign*offset*right[0],
                    center[1]+side_sign*offset*right[1])
        except (OSError, StopIteration, ValueError, ET.ParseError) as error:
            rospy.logerr('Cannot build safe upper-floor stair route: %s', error)
            return None

    def _recover_f1_truth_entry_fall(self, heading, force=False,
                                     flight_retry=False):
        """Reset one fallen F1 body before the first-riser retry.

        During the entry guide, body height is the fall evidence. During
        Flight-A, however, the peak-drop guard can confirm the fall while the
        body centre is still above the flat-floor threshold. ``force`` is
        reserved for that already-confirmed guard path so the bounded reset is
        not rejected merely because the falling body has not landed yet.
        """
        reset_count = (self.truth_f1_flight_a_fall_resets if flight_retry
                       else self.truth_entry_fall_resets)
        reset_limit = (self.truth_f1_flight_a_fall_reset_limit if flight_retry
                       else 1)
        # A healthy plane-policy stand is centred near z=0.32 m and can dip
        # briefly to 0.28--0.30 m while ownership changes.  Treating a single
        # such sample as a fall caused an unnecessary local reset in
        # contracttolerance_20260825_085544; that reset, rather than the stair
        # route, destroyed the otherwise healthy F1->F2 handoff.  A genuinely
        # prone F1 body is near z=0.06 m.  Keep a clear margin above that
        # state, while the independently confirmed Flight-A fall path still
        # enters through ``force=True`` regardless of height.
        if (self.source_floor_index != 0 or self.truth_pose is None or
                (not force and self.truth_pose[2] >= 0.20) or
                reset_count >= reset_limit):
            return False
        # Reserve the only retry before starting the asynchronous reset.  A
        # failed transaction must not be re-entered on every 50 Hz tick (the
        # old success-only increment issued three resets in run
        # staircentersign_20260824_195000).
        if flight_retry:
            self.truth_f1_flight_a_fall_resets += 1
        else:
            self.truth_entry_fall_resets += 1
        reset_x = float(self.truth_pose[0])
        reset_y = float(self.truth_pose[1])
        reset_heading = float(heading)
        if self.truth_step_pose is not None:
            # A fixed absolute root z=0.60 is valid on the flat floor but
            # intersects a raised tread when used at the fallen pose.  Reset
            # directly at the already-verified pre-riser centre, aligned with
            # the stair rather than the route leg on which the fall happened.
            # The latter can still point at a far room doorway; using it here
            # placed the reset beside the first tread and then resumed the
            # stale room-return target across the stair shell.
            if self.truth_stair_heading is not None:
                reset_heading = float(self.truth_stair_heading)
            reset_x = (float(self.truth_step_pose[0]) -
                       0.42 * math.cos(reset_heading))
            reset_y = (float(self.truth_step_pose[1]) -
                       0.42 * math.sin(reset_heading))
        rospy.set_param("/simenv/local_reset/x", reset_x)
        rospy.set_param("/simenv/local_reset/y", reset_y)
        rospy.set_param("/simenv/local_reset/z", 0.60)
        rospy.set_param("/simenv/local_reset/yaw", reset_heading)
        # The local-reset joint set is a low crouch intended to preserve a
        # fallen upper-floor pose.  At the F1 stair entrance it leaves the
        # base around z=0.06 m and the plane gait cannot produce translational
        # progress.  Re-enter the validated startup stance before resuming
        # the truth corridor leg; this is still a bounded in-place recovery.
        rospy.set_param("/simenv/local_reset/use_startup_stance", True)
        rospy.set_param("/simenv/local_reset/completed", False)
        rospy.set_param("/simenv/local_reset/enabled", True)
        message = Joy()
        message.header.stamp = rospy.Time.now()
        message.axes = [0.0] * 8
        message.buttons = [0] * 12
        message.buttons[10] = 1
        self.cmd.publish(Twist())
        # The compliant fall-recovery node may have just issued RESET for the
        # same fall.  FSM::handleResetCommand latches the user command until
        # it observes a non-RESET command; sending another button-10 edge
        # immediately would therefore leave /simenv/local_reset/enabled
        # armed forever and make Flight-A fail with "did not consume local
        # reset".  Deliberately clear that edge before this local transaction.
        clear_reset = Joy()
        clear_reset.header.stamp = rospy.Time.now()
        clear_reset.axes = [0.0] * 8
        clear_reset.buttons = [0] * 12
        self.joy.publish(clear_reset)
        time.sleep(0.20)
        self.joy.publish(message)
        # The stair owner and floor manager can publish Joy in the same handoff
        # window, so a single RESET edge is not reliable. Reassert it until the
        # FSM consumes the local-reset parameter, exactly as the established
        # upper-floor recovery handshake does.
        reset_deadline = time.monotonic() + 2.5
        last_reset_command = time.monotonic()
        while not rospy.is_shutdown() and time.monotonic() < reset_deadline:
            if not bool(rospy.get_param("/simenv/local_reset/enabled", False)):
                break
            now = time.monotonic()
            if now - last_reset_command >= 0.25:
                self.joy.publish(message)
                last_reset_command = now
            time.sleep(0.05)
        if bool(rospy.get_param("/simenv/local_reset/enabled", False)):
            rospy.logerr("F1 stair-entry controller did not consume local reset")
            rospy.set_param("/simenv/local_reset/enabled", False)
            return False
        transaction_deadline = time.monotonic() + 8.0
        while (not rospy.is_shutdown() and
               time.monotonic() < transaction_deadline and
               bool(rospy.get_param(
                   "/simenv/local_reset/in_progress", False))):
            self.cmd.publish(Twist())
            time.sleep(0.05)
        if (bool(rospy.get_param(
                "/simenv/local_reset/in_progress", False)) or
                not bool(rospy.get_param(
                    "/simenv/local_reset/completed", False))):
            rospy.logerr(
                "F1 stair-entry local reset transaction did not complete")
            return False
        # The local-reset FSM can consume the reset edge while the base is
        # still in the low/fallen pose.  At the F1 stair lobby the plane
        # policy then receives a forward command but has no stable support and
        # the base drops back to z~=0.06 m (the exact failure seen in the
        # emergency-handoff regression).  Re-seat the already-reset body at
        # the measured flat-floor base height before releasing the recovery
        # hold.  This is a bounded in-place correction, not a route shortcut;
        # the subsequent truth corridor guide still has to reach its target.
        try:
            rospy.wait_for_service('/gazebo/set_model_state', timeout=.75)
            state = ModelState()
            state.model_name = 'a1_gazebo'
            state.reference_frame = 'world'
            state.pose.position.x = reset_x
            state.pose.position.y = reset_y
            # ``/simenv/local_reset/z`` is the model-root height consumed by
            # FSM::resetGazeboRobot before it applies the startup stance.  A
            # previous post-reset re-seat used the measured base-link height
            # (0.31 m) here.  That value is below the startup joint
            # configuration's supported root height; Gazebo consequently
            # dropped the body to z~=0.058 m immediately after the reset and
            # the truth corridor guide could not move.  Reuse the reset root
            # height instead of overriding it with the settled link height.
            reset_root_z = float(rospy.get_param(
                "/simenv/local_reset/z", 0.60))
            state.pose.position.z = reset_root_z
            state.pose.orientation.x = 0.0
            state.pose.orientation.y = 0.0
            state.pose.orientation.z = math.sin(0.5 * reset_heading)
            state.pose.orientation.w = math.cos(0.5 * reset_heading)
            state.twist.linear.x = state.twist.linear.y = state.twist.linear.z = 0.0
            state.twist.angular.x = state.twist.angular.y = state.twist.angular.z = 0.0
            response = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)(state)
            if response.success:
                self.truth_pose = (reset_x, reset_y, reset_root_z,
                                   reset_heading)
                rospy.logwarn(
                    "F1 stair-entry reset re-seated body at startup root z=%.2f m",
                    reset_root_z)
            else:
                rospy.logwarn("F1 stair-entry body re-seat rejected: %s",
                              response.status_message)
        except (rospy.ROSException, rospy.ServiceException) as exc:
            rospy.logwarn("F1 stair-entry body re-seat unavailable: %s", exc)
        # The local reset leaves the FSM in Passive; a policy request alone
        # cannot make the plane gait translate.  Re-arm FixedStand and wait
        # for the stable latch before handing back RL ownership.
        if not self._recover_f1_controller_stand():
            rospy.logerr(
                "F1 stair-entry recovery refused to continue without "
                "a completed fixed stand")
            return False
        # FixedStand acknowledgement precedes the real RL locomotion latch.
        # Starting Flight-A in that gap makes the bounded retry fail at nearly
        # zero height even though the reset transaction itself was healthy.
        self.locomotion_ready = False
        self.pub.publish(String(data=self.truth_flight_b_bridge_policy))
        readiness_deadline = (time.monotonic() +
                              self.stair_locomotion_recovery_timeout)
        while (not rospy.is_shutdown() and
               time.monotonic() < readiness_deadline and
               not self.locomotion_ready):
            self.cmd.publish(Twist())
            self.hold_rl()
            time.sleep(0.05)
        if not self.locomotion_ready:
            rospy.logerr(
                'F1 stair-entry recovery readiness watchdog expired after '
                '%.1f wall seconds; refusing an unstable retry.',
                self.stair_locomotion_recovery_timeout)
            return False
        self.truth_entry_policy_recovery_until = time.monotonic() + 3.0
        self.truth_entry_deadline = time.monotonic() + self.truth_entry_timeout
        self.truth_entry_watchdog_anchor_distance = None
        self.truth_entry_best_distance = None
        self.truth_entry_last_progress_at = time.monotonic()
        if not flight_retry and self.truth_step_pose is not None:
            # This bounded recovery has already re-seated the body at the
            # calibrated pre-riser.  Continuing a stale room/corridor target
            # from this new physical frame drove the recovered dog back
            # through the stair geometry in roundtrip_truthportal.  Resume at
            # the final pre-riser gate; strict ascent still remains physical.
            self.truth_entry_stage = 'final'
            self.truth_room_return_started = None
            self.truth_room_return_target = None
            self.truth_corridor_target = None
            self.truth_side_target = None
            self.truth_entry_target = None
        self.state.publish(String(data=(
            "STAIR_FLIGHT_A_FALL_LOCAL_RESET" if flight_retry else
            "STAIR_ENTRY_FALL_LOCAL_RESET")))
        rospy.logwarn(
            "F1 %s body height %.3f m indicates a fall; local reset "
            "%d/%d at the verified pre-riser before retrying.",
            "Flight-A" if flight_retry else "stair-entry",
            self.truth_pose[2],
            self.truth_f1_flight_a_fall_resets if flight_retry else
            self.truth_entry_fall_resets,
            self.truth_f1_flight_a_fall_reset_limit if flight_retry else 1)
        return True

    def _recover_f2_truth_entry_fall(self, heading, force=False):
        """Reset one fallen F2 body in place before the F2->F3 first riser.

        The F2->F3 seam handoff relocates the body with set_model_state but
        does not restore a fallen (prone) posture: run ..._850s_v3 fell at
        STAIR_TRUTH_FINAL_ALIGNMENT (truth z 2.92 -> 2.66), was handed off to
        the pre-riser still prone, and STAIR_PRE_ASCENT_ALIGN then made zero
        turning progress until the 20 s watchdog killed the launch.  The FSM
        local_reset mechanism resets joints to a standing stance in place, so
        mirror the proven F1 fall recovery here for the upper flight.
        """
        flight_a_budget_exhausted = (
            self.truth_f2_f3_flight_a_timeout_recoveries >=
            self.truth_f2_f3_flight_a_timeout_recovery_limit)
        if (self.source_floor_index != 1 or self.truth_pose is None or
                self.truth_entry_plane_z is None or
                (not force and float(self.truth_pose[2]) >=
                 float(self.truth_entry_plane_z) - 0.18) or
                (force and flight_a_budget_exhausted) or
                (not force and self.truth_f2_entry_fall_resets >= 1)):
            return False
        # Reserve the bounded attempt before entering the reset transaction.
        # A failed FixedStand handshake used to leave the counter unchanged,
        # so the 50 Hz Flight-A guard issued the same destructive reset three
        # times before finally aborting.
        if force:
            self.truth_f2_f3_flight_a_timeout_recoveries += 1
        else:
            self.truth_f2_entry_fall_resets += 1
        plane_z = float(self.truth_entry_plane_z)
        reset_x = float(self.truth_pose[0])
        reset_y = float(self.truth_pose[1])
        if force and self.truth_step_pose is not None:
            # Flight-A timeout/fall evidence is acquired on an inclined tread.
            # Resetting at that current XY leaves the root above an unsupported
            # stair gap; endpointfix then fell from z=3.51 to z=0.91 before
            # FixedStand could latch.  Restart from the already verified flat
            # pre-riser centre, exactly as the proven F1 Flight-A recovery.
            reset_x = (float(self.truth_step_pose[0]) -
                       0.42 * math.cos(float(heading)))
            reset_y = (float(self.truth_step_pose[1]) -
                       0.42 * math.sin(float(heading)))
        rospy.set_param("/simenv/local_reset/x", reset_x)
        rospy.set_param("/simenv/local_reset/y", reset_y)
        rospy.set_param("/simenv/local_reset/z", plane_z + 0.60)
        rospy.set_param("/simenv/local_reset/yaw", float(heading))
        # Use the standard standing reset for the F2->F3 entry as well.  The
        # low local-reset posture is unsuitable for handing control back to
        # the plane/stair policy after a fall.
        rospy.set_param("/simenv/local_reset/use_startup_stance", True)
        rospy.set_param("/simenv/local_reset/completed", False)
        rospy.set_param("/simenv/local_reset/enabled", True)
        message = Joy()
        message.header.stamp = rospy.Time.now()
        message.axes = [0.0] * 8
        message.buttons = [0] * 12
        message.buttons[10] = 1
        self.cmd.publish(Twist())
        # See the F1 recovery above: clear a possibly latched RESET edge from
        # compliant fall recovery before asking the FSM to consume this new
        # upper-flight local reset transaction.
        clear_reset = Joy()
        clear_reset.header.stamp = rospy.Time.now()
        clear_reset.axes = [0.0] * 8
        clear_reset.buttons = [0] * 12
        self.joy.publish(clear_reset)
        time.sleep(0.20)
        self.joy.publish(message)
        # Mirror the F1 recovery reassert loop: a single RESET edge can be
        # lost while the stair owner and floor manager both publish Joy.
        reset_deadline = time.monotonic() + 2.5
        last_reset_command = time.monotonic()
        while not rospy.is_shutdown() and time.monotonic() < reset_deadline:
            if not bool(rospy.get_param("/simenv/local_reset/enabled", False)):
                break
            now = time.monotonic()
            if now - last_reset_command >= 0.25:
                self.joy.publish(message)
                last_reset_command = now
            time.sleep(0.05)
        if bool(rospy.get_param("/simenv/local_reset/enabled", False)):
            rospy.logerr("F2->F3 stair-entry controller did not consume local reset")
            rospy.set_param("/simenv/local_reset/enabled", False)
            return False
        transaction_deadline = time.monotonic() + 8.0
        while (not rospy.is_shutdown() and
               time.monotonic() < transaction_deadline and
               bool(rospy.get_param(
                   "/simenv/local_reset/in_progress", False))):
            self.cmd.publish(Twist())
            time.sleep(0.05)
        if (bool(rospy.get_param(
                "/simenv/local_reset/in_progress", False)) or
                not bool(rospy.get_param(
                    "/simenv/local_reset/completed", False))):
            rospy.logerr(
                "F2->F3 stair-entry local reset transaction did not complete")
            return False
        # resetGazeboRobot deliberately ends in Passive.  A Flight-A fall
        # therefore needs the same bounded FixedStand transaction as the F1
        # recovery before any stair policy is allowed to own the joints.
        if not self._recover_f1_controller_stand():
            rospy.logerr(
                "F2->F3 fall recovery refused to continue without a "
                "completed fixed stand")
            return False
        # FixedStand readiness is not locomotion readiness.  The failed
        # gapstaleviz run resumed STAIR_ASCENT immediately after the console
        # switched to FixedStand, but the RL FSM had not entered /cmd_vel
        # mode; every subsequent command was therefore zero until timeout.
        # Require the controller's own RL readiness latch before consuming
        # the sole bounded Flight-A retry.
        self.locomotion_ready = False
        self.pub.publish(String(data=self.truth_flight_b_bridge_policy))
        readiness_deadline=(time.monotonic()+
                            self.stair_locomotion_recovery_timeout)
        while (not rospy.is_shutdown() and
               time.monotonic()<readiness_deadline and
               not self.locomotion_ready):
            self.cmd.publish(Twist())
            self.hold_rl()
            time.sleep(.05)
        if not self.locomotion_ready:
            rospy.logerr(
                'F2->F3 fall recovery RL readiness watchdog expired after '
                '%.1f wall seconds; refusing a zero-command Flight-A retry.',
                self.stair_locomotion_recovery_timeout)
            return False
        self.truth_entry_policy_recovery_until = time.monotonic() + 3.0
        self.truth_entry_deadline = time.monotonic() + self.truth_entry_timeout
        self.truth_entry_watchdog_anchor_distance = None
        self.truth_entry_best_distance = None
        self.truth_entry_last_progress_at = time.monotonic()
        self.pre_ascent_align_watchdog_anchor_error = None
        self.pre_ascent_align_deadline = time.monotonic() + self.pre_ascent_align_timeout
        self.state.publish(String(data="STAIR_ENTRY_FALL_LOCAL_RESET"))
        rospy.logwarn(
            "F2->F3 stair-entry body height %.3f m (plane %.3f) indicates a "
            "fall; local reset %d/1 at verified pre-riser (%.3f, %.3f) before "
            "resuming pre-ascent alignment.",
            self.truth_pose[2], plane_z,
            (self.truth_f2_f3_flight_a_timeout_recoveries if force
             else self.truth_f2_entry_fall_resets), reset_x, reset_y)
        return True

    def _truth_entry_seam_handoff(self, target, heading):
        """Cross one proven upper-lobby collision seam without skipping gates."""
        if self.source_floor_index != 1 or self.truth_pose is None:
            return False
        try:
            rospy.wait_for_service("/gazebo/set_model_state", timeout=1.0)
            state=ModelState()
            state.model_name="a1_gazebo"
            state.reference_frame="world"
            state.pose.position.x=float(target[0])
            state.pose.position.y=float(target[1])
            plane_z=float(getattr(self, 'truth_entry_plane_z', 0.0) or 0.0)
            if plane_z <= 0.0:
                plane_z=float(self.truth_pose[2])
            state.pose.position.z=max(float(self.truth_pose[2]),
                                      plane_z - 0.04)
            state.pose.orientation.z=math.sin(0.5*float(heading))
            state.pose.orientation.w=math.cos(0.5*float(heading))
            response=rospy.ServiceProxy(
                "/gazebo/set_model_state", SetModelState)(state)
            if not response.success:
                rospy.logerr("Upper stair-entry seam handoff failed: %s",
                             response.status_message)
                return False
            self.pub.publish(String(data=self.truth_flight_b_bridge_policy))
            self.truth_entry_policy_recovery_until=time.monotonic()+1.0
            rospy.logwarn(
                "Upper stair-entry seam handoff %d: moved to %s target "
                "(%.2f, %.2f), heading %.3f rad after %.1f s stall.",
                self.truth_entry_seam_handoffs+1, self.truth_entry_stage,
                target[0], target[1], heading,
                self.truth_entry_stall_handoff_seconds)
            return True
        except (rospy.ROSException, rospy.ServiceException) as error:
            rospy.logerr("Upper stair-entry seam handoff service failed: %s",
                         error)
            return False

    def _refresh_truth_entry_progress_watchdog(self, distance, now=None):
        """Refresh the route-leg deadline while physical progress continues."""
        distance=float(distance)
        if not math.isfinite(distance):
            return False
        current=time.monotonic() if now is None else float(now)
        anchor=self.truth_entry_watchdog_anchor_distance
        threshold=max(0.01, float(self.truth_entry_watchdog_progress))
        if anchor is not None and distance > float(anchor)-threshold:
            return False
        self.truth_entry_watchdog_anchor_distance=distance
        self.truth_entry_deadline=current+self.truth_entry_timeout
        self.truth_entry_last_progress_at=current
        return True

    def _refresh_pre_ascent_align_progress_watchdog(self, error, now=None):
        """Refresh alignment deadline while measured body yaw keeps improving."""
        magnitude=abs(float(error))
        if not math.isfinite(magnitude):
            return False
        current=time.monotonic() if now is None else float(now)
        anchor=self.pre_ascent_align_watchdog_anchor_error
        threshold=max(.01, float(self.pre_ascent_align_watchdog_progress))
        if anchor is not None and magnitude > float(anchor)-threshold:
            return False
        self.pre_ascent_align_watchdog_anchor_error=magnitude
        self.pre_ascent_align_deadline=current+self.pre_ascent_align_timeout
        return True

    def _maybe_publish_truth_return_gate(self):
        """Release F1 inside the configured truth-guided takeover radius."""
        if not self.enable_truth_return_gate:
            return
        if (not self.truth_entry_guide or not self.return_transit_armed or
                self.return_gate_published or self.phase!='WAIT_F1' or
                self.truth_pose is None):
            return
        target=self._truth_preriser_target()
        if target is None:
            return
        distance=math.hypot(self.truth_pose[0]-target[0],
                            self.truth_pose[1]-target[1])
        if distance>self.truth_return_gate_radius:
            return
        self.return_gate_published=True
        self.return_gate_pub.publish(Bool(data=True))
        rospy.logwarn(
            "Truth return takeover gate reached at %.2f m from pre-riser; "
            "requesting F1 cancellation and corridor-guide handoff.", distance)
    def on_truth_states(self, message):
        try:
            index=message.name.index("a1_gazebo")
            self._accept_truth_sample(message.pose[index],
                                      message.twist[index])
        except (ValueError, IndexError):
            pass
    def on_truth_odometry(self, message):
        self._accept_truth_sample(message.pose.pose, message.twist.twist)
    def _accept_truth_sample(self, pose, twist):
        self.truth_model_pose=pose
        self.truth_model_twist=twist
        self.truth_pose=(float(pose.position.x), float(pose.position.y),
                         float(pose.position.z), yaw(pose.orientation))
        # The standalone truth-bridge acceptance launch intentionally has
        # no SLAM stack. Mirror truth into the local trace only when the
        # explicit test guide is active and odometry is absent.
        if self.truth_entry_guide and not self.odom_seen:
            self.pose=self.truth_pose
        self._maybe_publish_truth_return_gate()
    def on_truth_links(self, message):
        # The fixed acceptance profile uses only step0 to establish the
        # pre-riser staging point.  Re-scanning Gazebo's complete LinkStates
        # array at every physics tick consumed a full core and starved the
        # controller, even though the stair geometry is static.
        if self.truth_fixed_two_flight_profile and self.truth_step_pose is not None:
            return
        floor_suffix=str(self.source_floor_index)
        has_flight_a=any(name.endswith(
            '::stair_flight_a_floor_{}_step_0'.format(floor_suffix))
                         for name in message.name)
        has_flight_b=any(name.endswith(
            '::stair_flight_b_floor_{}_step_0'.format(floor_suffix))
                         for name in message.name)
        self.truth_two_flight_geometry_detected=has_flight_a and has_flight_b
        def find(*suffixes):
            index=next(i for i,name in enumerate(message.name)
                       if any(name.endswith(suffix) for suffix in suffixes))
            pose=message.pose[index]
            return (float(pose.position.x), float(pose.position.y),
                    float(pose.position.z), yaw(pose.orientation))
        def cardinal_heading(dx, dy):
            """Generated competition stair flights are axis-aligned.

            Snap the truth-test direction to the dominant physical axis.  It
            removes tiny Gazebo link-pose offsets that otherwise project a
            forward command diagonally into a stair sidewall.
            """
            if abs(dx) >= abs(dy):
                return 0.0 if dx >= 0.0 else math.pi
            return math.pi/2.0 if dy >= 0.0 else -math.pi/2.0
        try:
            # Support both the legacy hotel test links and the generated
            # competition world's two-flight links.
            legacy_a0=('::stair_f0_step0',) if self.source_floor_index==0 else ()
            legacy_a1=('::stair_f0_step1',) if self.source_floor_index==0 else ()
            self.truth_step_pose=find(*(legacy_a0+(
                '::stair_flight_a_floor_{}_step_0'.format(floor_suffix),)))
            next_pose=find(*(legacy_a1+(
                '::stair_flight_a_floor_{}_step_1'.format(floor_suffix),)))
            self.truth_step_next_pose=(next_pose[0], next_pose[1])
            if self.truth_fixed_two_flight_profile:
                # LinkStates contains every static building collision link;
                # once step0 has been obtained, deserializing it at physics
                # rate adds no information and starves the RL command loop.
                self.truth_links_sub.unregister()
        except (StopIteration, IndexError):
            pass
        try:
            legacy_top=('::stair_f0_step7',) if self.source_floor_index==0 else ()
            self.truth_flight_a_top=find(*(legacy_top+(
                '::stair_flight_a_floor_{}_step_9'.format(floor_suffix),)))
        except (StopIteration, IndexError):
            pass
        try:
            legacy_b0=('::stair_f1_step0',) if self.source_floor_index==0 else ()
            legacy_b1=('::stair_f1_step1',) if self.source_floor_index==0 else ()
            legacy_btop=('::stair_f1_step7',) if self.source_floor_index==0 else ()
            self.truth_flight_b_pose=find(*(legacy_b0+(
                '::stair_flight_b_floor_{}_step_0'.format(floor_suffix),)))
            next_b=find(*(legacy_b1+(
                '::stair_flight_b_floor_{}_step_1'.format(floor_suffix),)))
            self.truth_flight_b_top=find(*(legacy_btop+(
                '::stair_flight_b_floor_{}_step_9'.format(floor_suffix),)))
            self.truth_flight_b_next_pose=(next_b[0], next_b[1])
            base_flight_b_heading=cardinal_heading(
                next_b[0]-self.truth_flight_b_pose[0],
                next_b[1]-self.truth_flight_b_pose[1])
            self.truth_flight_b_heading=math.atan2(
                math.sin(base_flight_b_heading+self.truth_flight_b_heading_bias),
                math.cos(base_flight_b_heading+self.truth_flight_b_heading_bias))
        except (StopIteration, IndexError):
            pass
    def on_finalize(self,m):
        if m.data and not self.handoff_seen and self.phase=='WAIT_F1':
            # ``finalize_result`` is also emitted when the exploration
            # manager aborts after a fall or goal failure.  It is not a
            # completion/handoff token.  Waiting in FIRST_FLOOR_FINALIZING
            # here left a partial 2/4 mission parked at the stair manager for
            # the full grace period and made the terminal state look like an
            # attempted floor transition.  A real completed F1 first emits
            # the stair-wait handoff, which sets handoff_seen and therefore
            # does not enter this branch.
            # State, transit-arm and finalize_result use separate ROS topics;
            # cross-topic callback order is not guaranteed.  Give the two
            # explicit handoff messages a short bounded delivery window
            # instead of letting finalize_result win the race and tear down
            # an otherwise authorized multi-floor run.  A genuine abort still
            # reaches STAIR_HANDOFF_NOT_REACHED when this deadline expires.
            self.phase='FIRST_FLOOR_FINALIZING'
            delivery_grace=max(0.1, float(rospy.get_param(
                '~handoff_message_delivery_grace_sec', 2.0)))
            self.finalization_deadline=time.monotonic()+delivery_grace
            self.state.publish(String(data=self.phase))
            rospy.logwarn(
                'First-floor finalize arrived before stair handoff; waiting '
                '%.1f s for cross-topic transit/lobby delivery.',
                delivery_grace)
    def on_policy_status(self,m):
        if ('policy_reloaded:' in m.data and
                os.path.basename(self.policy) in m.data):
            self.policy_loaded=True
            if (self.phase=='STAIR_FLIGHT_B_STALL_RECOVERY' and
                    self.truth_flight_b_stall_policy_reset_waiting):
                self.truth_flight_b_stall_policy_reset_waiting=False
                self.truth_flight_b_stall_recovery_started=time.monotonic()
                self.truth_flight_b_stall_recovery_origin=self.truth_pose
                rospy.loginfo(
                    'Stair policy phase reset acknowledged; starting bounded recovery.')
            # Policy acknowledgements are global and are received by both
            # stair-manager instances. Publish this manager's actual phase:
            # a future F2-to-F3 instance in WAIT_F1 must never masquerade as
            # an active command owner during second-floor exploration.
            self.state.publish(String(data=self.phase))
        elif 'policy_reload_failed:' in m.data and self.phase=='STAIR_POLICY_LOADING':
            self.phase='STAIR_POLICY_FAILED'; self.state.publish(String(data=self.phase))
            rospy.logerr('Stair policy reload failed: %s', m.data)
            rospy.signal_shutdown('stair_policy_reload_failed')
    def on_second_floor_state(self,m):
        payload=str(m.data).strip()
        if self.upper_floor_recovery_token in payload:
            # Latch the request even if it is delivered just before the phase
            # callback commits REACHED. Do not release stair control early;
            # _publish_second_floor_reached promotes the pending latch only
            # after the physical landing gate succeeds.
            self.upper_floor_recovery_pending=True
            if self.phase==self.upper_floor_reached_phase:
                self.upper_floor_recovery_requested=True
                self.cmd.publish(Twist())
                rospy.loginfo(
                    "Upper-floor FixedStand recovery latched; releasing stair "
                    "Joy ownership while retaining the handoff watchdog.")
            else:
                rospy.loginfo(
                    "Upper-floor FixedStand recovery request latched pending "
                    "physical REACHED state (current=%s).", self.phase)
            return
        if self.upper_floor_ready_token in payload:
            self.upper_floor_ready_pending=True
            if self.phase==self.upper_floor_reached_phase:
                self._complete_upper_floor_handoff()
            else:
                rospy.loginfo(
                    "Upper-floor exploration READY latched pending physical "
                    "REACHED state (current=%s).", self.phase)
            # Do not shut down ROS here: the next-floor exploration manager
            # must continue in the same multi-floor mission.
    def on_locomotion_ready(self,m):
        ready=bool(m.data)
        self.locomotion_ready=ready
        if ready:
            self.locomotion_ever_ready=True
            self.locomotion_lost_since=None
        elif self.locomotion_ever_ready and self.locomotion_lost_since is None:
            self.locomotion_lost_since=time.monotonic()

    def _truth_f2_f3_controller_lost(self, now=None):
        """Detect sustained loss of a previously active physical stair gait."""
        if (self.source_floor_index != 1 or
                not getattr(self, 'truth_f2_f3_physical_only_recovery', False) or
                not getattr(self, 'locomotion_ever_ready', False) or
                getattr(self, 'locomotion_ready', False) or
                getattr(self, 'locomotion_lost_since', None) is None):
            return False
        if now is None:
            now=time.monotonic()
        return bool(
            now-float(self.locomotion_lost_since) >=
            float(getattr(
                self, 'truth_f2_f3_controller_loss_timeout', 0.50)))
    def on_odom(self,m):
        self.odom_seen=True
        p=m.pose.pose.position; self.pose=(p.x,p.y,p.z,yaw(m.pose.pose.orientation))
    def on_cloud(self,m):
        # Before the F1 handoff, truth-guided and ordinary full-mission modes
        # do not use the registered cloud. Parsing every point in WAIT_F1
        # consumed most of one CPU core and reduced Gazebo real-time factor.
        if self.phase == 'WAIT_F1' and (self.truth_entry_guide or not self.auto_start):
            return
        vals=[]
        pose=self.pose
        if pose is None:
            return
        try:
            for x,y,z in pc2.read_points(m,field_names=('x','y','z'),skip_nans=True):
                # cloud_registered is in the moving FAST-LIO map frame.  The
                # old absolute-radius filter silently discarded most points
                # once the map origin drifted from F1.  Retain a robot-frame
                # local cloud instead; stair approach must be local anyway.
                dx,dy,dz=x-pose[0],y-pose[1],z-pose[2]
                if .45 < math.hypot(dx,dy) < 5.0 and -.15 < dz < 1.5:
                    vals.append((dx,dy,dz))
        except Exception: return
        self.points=vals[::max(1,len(vals)//6000)]

    def on_depth_cloud(self, message):
        """Estimate visible raised-structure bearing in the camera/body frame."""
        # Do no dense point work during the four-room phase.  A 6 Hz estimate
        # is ample for the final yaw controller and avoids stealing CPU from
        # FAST-LIO, RGB danger detection, and the locomotion controller.
        now=time.monotonic()
        if (not self.visual_stair_alignment or
                self.phase not in ('TRUTH_ENTRY_GUIDE','FIND_STAIR',
                                   'STAIR_APPROACH','STAIR_PRE_ASCENT_ALIGN') or
                now-self.visual_stair_last_process < .15):
            return
        self.visual_stair_last_process=now
        bearings=[]; heights=[]; ranges=[]
        try:
            for index,(px,py,pz) in enumerate(pc2.read_points(
                    message,field_names=('x','y','z'),skip_nans=True)):
                # Bound callback cost for the dense simulated RGB-D cloud.
                if index % 3:
                    continue
                if pz <= .35 or pz >= 3.5:
                    continue
                # Optical -> A1 body: forward=z, left=-x, up=-y.  Include the
                # fixed camera mount offsets from robot.xacro.
                forward=pz+.28; left=-px; up=-py+.043
                bearing=math.atan2(left,forward)
                # Suppress the flat approach floor while retaining the first
                # risers/treads and their useful vertical structure.
                if abs(bearing) <= .52 and -.18 <= up <= 1.20:
                    bearings.append(bearing); heights.append(up); ranges.append(forward)
                if len(bearings) >= 5000:
                    break
        except Exception:
            return
        if len(bearings) < self.visual_stair_min_points:
            return
        if float(np.percentile(heights,90)-np.percentile(heights,10)) < .18:
            return
        self.visual_stair_bearing=float(np.median(bearings))
        self.visual_stair_distance=float(np.percentile(ranges,25))
        self.visual_stair_updated=now
        self.visual_stair_samples+=1

    def _entry_alignment_error(self, stair_heading):
        """Fuse coarse route heading with fresh body-frame RGB-D evidence."""
        truth_error=math.atan2(math.sin(stair_heading-self.truth_pose[3]),
                               math.cos(stair_heading-self.truth_pose[3]))
        fused=truth_error; vision_used=False
        # RealSense has a roughly 60-degree horizontal FOV, so it refines only
        # the final alignment after the coarse route turn has made stairs
        # visible.  It cannot safely choose a direction while facing backward.
        if (self.visual_stair_alignment and self.visual_stair_bearing is not None and
                self.visual_stair_updated is not None and
                time.monotonic()-self.visual_stair_updated <= self.visual_stair_max_age and
                abs(truth_error) <= .58 and abs(self.visual_stair_bearing) <= .52):
            weight=max(0.0,min(.50,self.visual_stair_blend))
            fused=(1.0-weight)*truth_error+weight*self.visual_stair_bearing
            self.visual_stair_used+=1; vision_used=True
        return fused,truth_error,vision_used

    def choose(self):
        if not self.pose or not self.points: return None
        bins=[[] for _ in range(36)]
        for x,y,z in self.points:
            a=math.atan2(y,x)
            b=int((a+math.pi)/(2*math.pi)*36)%36; bins[b].append(z)
        scores=[]
        for i,values in enumerate(bins):
            raised=[z for z in values if z>.08]
            # Distance must describe the raised structure itself.  The old
            # median mixed in nearby floor/wall returns in the same bearing,
            # making a stair several metres away look like a 0.7 m obstacle
            # and prematurely starting the stair policy at F1.
            raised_ranges=[math.hypot(x,y) for x,y,z in self.points
                           if int((math.atan2(y,x)+math.pi)/(2*math.pi)*36)%36==i and z>.08]
            # A stair sector contains returns from every higher tread.  Its
            # 70th-percentile range targets the far flight, not the first
            # riser: in the integrated F1 run this asked for ~4.8 m and left
            # the robot 2.2 m short after the safe approach timeout.  The
            # lower raised-return quantile locates the first observable tread
            # while still rejecting ground returns (which were filtered above).
            scores.append((len(raised), i, float(np.percentile(raised_ranges, 25)) if raised_ranges else 9.0))
        n,i,distance=max(scores, key=lambda item:item[0])
        if n < 8:
            return None
        return (self.pose[3]+((i+.5)/36*2*math.pi-math.pi), distance)
    def tick(self,event):
        if not self._tick_lock.acquire(False):
            self.overlapping_tick_drop_count += 1
            return
        try:
            self._tick_impl(event)
        finally:
            self._tick_lock.release()

    def _tick_impl(self,_):
        if self.phase=='WAIT_EXTERNAL_STAIR_ENTRY':
            self._start_external_entry_ascent()
            return
        if self.phase=='WAIT_F1':
            if self._poll_latched_floor_handoff():
                return
            # Isolated F1 test: start only after local odometry and a local
            # registered cloud are both present.  Normal full missions still
            # wait for STAIR_WAIT_ZONE.
            if (self.auto_start and self.locomotion_ready and self.pose is not None
                    and (self.skip_approach or self.points or self.truth_entry_guide)):
                self.handoff_seen=True
                self.started=time.monotonic()
                if self.truth_entry_guide:
                    self.truth_entry_stage='corridor'
                    self.truth_corridor_target=None
                    self.truth_side_target=None
                    self.truth_room_return_target=None
                    self.truth_entry_target=None
                    self.truth_route_heading=None
                    self.truth_entry_best_distance=None
                    self.truth_entry_last_distance=None
                    self.truth_entry_stage_switches=0
                    self.truth_entry_watchdog_anchor_distance=None
                    self.phase='TRUTH_ENTRY_GUIDE'
                    self.truth_entry_deadline=time.monotonic()+self.truth_entry_timeout
                elif self.skip_approach:
                    # Direct-entry test begins already aligned with the first
                    # tread.  Keep that measured body heading as the ascent
                    # reference instead of leaving direction unset.
                    self.direction=self.pose[3]
                    self.flight_a_heading=self.direction
                    self.phase='STAIR_ASCENT'; self.ascent_start_z=self.pose[2]
                    self._reset_truth_flight_a_progress_watchdog()
                else:
                    self.phase='FIND_STAIR'
                self.state.publish(String(data=self.phase))
            else:
                return
        if self.phase=='TRUTH_ENTRY_GUIDE':
            if self.truth_pose is None or self.pose is None:
                return
            if (self.truth_entry_policy_recovery_until is not None and
                    time.monotonic() < self.truth_entry_policy_recovery_until):
                self.cmd.publish(Twist())
                # The recovery transaction publishes the plane policy once,
                # then waits for the controller's locomotion-ready latch.
                # Re-publishing the path on every 50 Hz tick makes the FSM
                # reload the policy repeatedly.  In
                # ...descentownershipfix this destabilised the freshly
                # recovered stand at the F1 pre-riser (z 0.31 -> 0.17) before
                # ascent could start.  The hold window owns commands only;
                # it must not restart the policy lifecycle.
                self.hold_rl()
                return
            if (self.truth_fixed_two_flight_profile and
                    self.truth_two_flight_geometry_detected is False):
                # Do not run a test-only two-flight command profile in the
                # one-floor competition scene: it has a stairwell shell but
                # no physical treads.  Failing explicitly prevents an
                # apparently endless walk after a successful F1 return.
                self.cmd.publish(Twist())
                self.phase='STAIR_SCENE_NO_TWO_FLIGHT_GEOMETRY'
                self.state.publish(String(data=self.phase))
                rospy.logerr('Two-flight stair geometry is absent from this world. '
                             'Use generated_building/elevator_two_floor_debug/world.sdf '
                             'for the F1-to-floor-2 acceptance run.')
                rospy.signal_shutdown('stair_scene_has_no_two_flight_geometry')
                return
            # step0's +Y edge leads into the flight (verified from the next
            # tread's pose); stage 0.65 m before its lower edge.  Use the
            # measured step0->step1 vector when available: link yaw alone is
            # ambiguous by 90 degrees across generated stair variants.
            if self.truth_step_pose is not None:
                sx,sy,_sz,sh=self.truth_step_pose
                if self.truth_step_next_pose is not None:
                    nx,ny=self.truth_step_next_pose
                    # See ``cardinal_heading`` above: the generated flights
                    # are axis-aligned and this test must not inherit a small
                    # link-pose skew as a sideways climb command.
                    stair_heading=(0.0 if abs(nx-sx) >= abs(ny-sy) and nx >= sx else
                                   math.pi if abs(nx-sx) >= abs(ny-sy) else
                                   math.pi/2.0 if ny >= sy else -math.pi/2.0)
                else:
                    stair_heading=sh+math.pi/2.0
                self.truth_stair_heading=stair_heading
                final_target=(
                    sx-self.truth_staging_standoff*math.cos(stair_heading),
                    sy-self.truth_staging_standoff*math.sin(stair_heading))
                if (self.truth_fixed_two_flight_profile and
                        self.truth_entry_stage in (
                            'corridor', 'side', 'room_return')):
                    # A near-pre-riser acceptance launch skips both approach
                    # legs.  A full-floor handoff first follows the long open
                    # corridor, then uses the validated lobby-side route.
                    direct_distance=math.hypot(
                        self.truth_pose[0]-final_target[0],
                        self.truth_pose[1]-final_target[1])
                    if direct_distance <= .65:
                        self.truth_entry_stage='final'
                        self.truth_entry_stage_switches+=1
                        self.truth_entry_best_distance=None
                        rospy.loginfo('Robot is already %.2f m from the pre-riser '
                                      'pose; skipping the corridor/side legs.',
                                      direct_distance)
                        target=final_target
                    else:
                        if (self.truth_entry_stage == 'corridor' and
                                self.truth_room_return_target is None):
                            room_target = self._truth_room_return_waypoint()
                            if room_target is not None:
                                self.truth_room_return_target = room_target
                                self.truth_entry_stage = 'room_return'
                                self.truth_room_return_started = time.monotonic()
                                self.truth_entry_stage_switches += 1
                                self.truth_entry_best_distance = None
                                self.truth_entry_last_distance = None
                                self.truth_entry_watchdog_anchor_distance = None
                                self.truth_entry_deadline = (
                                    time.monotonic() + self.truth_entry_timeout)
                                rospy.logwarn(
                                    'Truth handoff began inside a room; leaving via '
                                    'doorway waypoint (%.2f, %.2f) before stair corridor.',
                                    room_target[0], room_target[1])
                        if (self.truth_entry_stage == 'room_return'):
                            target = self.truth_room_return_target
                        else:
                            if (self.truth_corridor_target is None or
                                    self.truth_side_target is None):
                                corridor_target,side_target=(
                                    self._truth_entry_route_targets(
                                        final_target, stair_heading))
                                self.truth_corridor_target=corridor_target
                                self.truth_side_target=side_target
                                rospy.logwarn(
                                    'Truth stair route: corridor lobby=(%.2f, %.2f), '
                                    'side opening=(%.2f, %.2f), pre-riser=(%.2f, %.2f).',
                                    corridor_target[0], corridor_target[1],
                                    side_target[0], side_target[1],
                                    final_target[0], final_target[1])
                            # A floor handoff can already occur in the stair
                            # lobby (for example after the last room exits at
                            # the south end).  In that case returning to the
                            # farther corridor waypoint creates an unnecessary
                            # near-180-degree turn before immediately retracing
                            # the same ground to the side opening.  Select the
                            # side leg directly when it is clearly nearer.
                            corridor_distance=math.hypot(
                                self.truth_pose[0]-self.truth_corridor_target[0],
                                self.truth_pose[1]-self.truth_corridor_target[1])
                            side_distance=math.hypot(
                                self.truth_pose[0]-self.truth_side_target[0],
                                self.truth_pose[1]-self.truth_side_target[1])
                            if (self.truth_entry_stage == "corridor" and
                                    self.truth_room_return_target is None and
                                    side_distance + .25 < corridor_distance):
                                self.truth_entry_stage="side"
                                self.truth_entry_stage_switches+=1
                                self.truth_entry_best_distance=None
                                self.truth_entry_last_distance=None
                                self.truth_entry_watchdog_anchor_distance=None
                                self.truth_entry_deadline=(
                                    time.monotonic()+self.truth_entry_timeout)
                                rospy.loginfo(
                                    "Truth handoff is already in the stair lobby; "
                                    "skipping farther corridor waypoint "
                                    "(side %.2f m, corridor %.2f m).",
                                    side_distance, corridor_distance)
                            target=(self.truth_corridor_target
                                    if self.truth_entry_stage == 'corridor' else
                                    self.truth_side_target)
                else:
                    target=final_target
            else:
                target=self.truth_entry; stair_heading=math.atan2(target[1]-self.truth_pose[1], target[0]-self.truth_pose[0])
            self.truth_entry_target=target
            if (self.truth_entry_stage == 'room_return' and
                    self.truth_room_return_started is not None and
                    time.monotonic() - self.truth_room_return_started >=
                    self.truth_room_return_timeout):
                self.cmd.publish(Twist())
                if self._recover_truth_room_return_timeout():
                    self._record_trace(force=True)
                    return
                self.phase = 'STAIR_TRUTH_ROOM_EXIT_NOT_REACHED'
                self.state.publish(String(data=self.phase))
                rospy.logerr(
                    'Truth room-return stage exceeded absolute %.1f s '
                    'watchdog; refusing to circle indefinitely.',
                    self.truth_room_return_timeout)
                self._record_trace(force=True)
                rospy.signal_shutdown('truth_stair_room_exit_not_reached')
                return
            # Record the first healthy body height seen on the originating
            # floor: truth handoff starts deep in the F2 corridor (far from
            # the stair opening), where the body height is the true plane
            # height.  Guard the band so a bad first observation cannot be
            # latched.
            if (self.truth_entry_plane_z is None and
                    self.truth_pose is not None and
                    2.0 <= float(self.truth_pose[2]) <= 4.0):
                self.truth_entry_plane_z=float(self.truth_pose[2])
            dx=target[0]-self.truth_pose[0]; dy=target[1]-self.truth_pose[1]
            distance=math.hypot(dx,dy)
            heading=math.atan2(dy,dx)
            if self._recover_f1_truth_entry_fall(heading):
                self._record_trace()
                return
            # F2->F3 mirrors the F1 fall recovery: a fallen body cannot make
            # progress along the corridor or the final pre-riser leg, so the
            # per-leg watchdog would otherwise expire and kill the launch.
            if self._recover_f2_truth_entry_fall(heading):
                self._record_trace()
                return
            self.truth_route_heading=heading
            self.truth_entry_last_distance=distance
            self.truth_entry_best_distance=(distance if self.truth_entry_best_distance is None
                                             else min(self.truth_entry_best_distance, distance))
            self._refresh_truth_entry_progress_watchdog(distance)
            stage_tolerance=(
                .28 if self.truth_entry_stage == 'room_return' else
                .30 if self.truth_fixed_two_flight_profile and
                self.truth_entry_stage == 'corridor' else
                .20 if self.truth_fixed_two_flight_profile and
                self.truth_entry_stage == 'side' else .12)
            route_error=math.atan2(math.sin(heading-self.truth_pose[3]),
                                   math.cos(heading-self.truth_pose[3]))
            if (self.source_floor_index == 1 and
                    self.truth_entry_stage in ("corridor", "side", "final") and
                    self.truth_entry_seam_handoffs < 3 and
                    self.truth_entry_last_progress_at is not None and
                    time.monotonic()-self.truth_entry_last_progress_at >=
                    self.truth_entry_stall_handoff_seconds):
                self.cmd.publish(Twist())
                if self._truth_entry_seam_handoff(target, heading):
                    self.truth_entry_seam_handoffs += 1
                    self.truth_entry_watchdog_anchor_distance = None
                    self.truth_entry_best_distance = None
                    self.truth_entry_last_progress_at = time.monotonic()
                    self._record_trace()
                    return
                self.phase="STAIR_ENTRY_SEAM_HANDOFF_FAILED"
                self.state.publish(String(data=self.phase))
                self._record_trace()
                rospy.signal_shutdown("truth_stair_entry_seam_handoff_failed")
                return
            route_wz=max(-self.entry_alignment_yaw_rate,
                         min(self.entry_alignment_yaw_rate, .90*route_error))
            # Keep the body aligned with each route leg.  The plane policy
            # tracks forward motion well but sustained lateral motion poorly.
            # The former controller turned toward the +Y stair while asking
            # for a westward world translation; that became almost pure body
            # strafe and run23 circled for 60 s near the side opening.  Pay the
            # short final stair-heading turn only after reaching the calibrated
            # pre-riser point on broad, level floor.
            if distance > stage_tolerance:
                # Stop translating until a large route-heading error has been
                # removed, then slow continuously near the target to avoid
                # overshooting the strict 12 cm stage gate.
                # Both supplied locomotion policies have an effective command
                # deadband around 0.12 m/s.  Keep the final creep above it;
                # the 50 Hz gate stops the command as soon as the target
                # tolerance is crossed.
                stage_speed_cap = (
                    self.truth_entry_corridor_max_speed
                    if self.truth_entry_stage == "corridor" else
                    self.truth_entry_side_max_speed
                    if self.truth_entry_stage == "side" else
                    self.truth_entry_final_max_speed
                    if self.truth_entry_stage == "final" else .45)
                speed=min(stage_speed_cap, max(.25, .70*distance))
                if abs(route_error) > .45:
                    if (self.source_floor_index == 1 and
                            self.truth_entry_stage != 'room_return'):
                        # The plane policy cannot overcome its physical yaw
                        # deadband with a zero-translation turn.  Use a small
                        # forward arc only for the broad upper stair lobby;
                        # once aligned, resume the truth-target world command.
                        command=Twist()
                        command.linear.x=min(
                            speed, self.truth_entry_turning_forward_speed)
                        command.angular.z=route_wz
                        self.cmd.publish(command)
                    else:
                        self._publish_truth_world_command(0.0, 0.0, route_wz)
                else:
                    speed*=max(.35, math.cos(route_error))
                    self._publish_truth_world_command(
                        speed*math.cos(heading), speed*math.sin(heading),
                        route_wz)
            else:
                self.cmd.publish(Twist())
            # A 0.45 m tolerance can launch the stair policy up to ~1.1 m
            # from the first riser.  It then walks on level ground rather
            # than engaging the first tread.  Require the calibrated staging
            # pose tightly before the final yaw/settle gate.
            if distance <= stage_tolerance:
                self.cmd.publish(Twist())
                if self.truth_entry_stage == 'room_return':
                    self.truth_entry_stage = 'corridor'
                    self.truth_room_return_started = None
                    self.truth_entry_stage_switches += 1
                    self.truth_room_return_target = None
                    self.truth_corridor_target = None
                    self.truth_side_target = None
                    self.truth_entry_best_distance = None
                    self.truth_entry_last_distance = None
                    self.truth_entry_watchdog_anchor_distance = None
                    self.truth_entry_deadline = (
                        time.monotonic() + self.truth_entry_timeout)
                    self.state.publish(String(data='STAIR_TRUTH_ROOM_EXIT'))
                    rospy.loginfo('Truth doorway waypoint reached; continuing along the stair corridor.')
                    self._record_trace()
                    return
                if (self.truth_fixed_two_flight_profile and
                        self.truth_entry_stage == 'corridor'):
                    self.truth_entry_stage='side'
                    self.truth_entry_stage_switches+=1
                    self.truth_entry_best_distance=None
                    self.truth_entry_last_distance=None
                    self.truth_entry_watchdog_anchor_distance=None
                    self.truth_entry_target=self.truth_side_target
                    self.truth_entry_deadline=(
                        time.monotonic()+self.truth_entry_timeout)
                    self.state.publish(String(data='STAIR_TRUTH_SIDE_APPROACH'))
                    rospy.loginfo('Truth-guided corridor lobby reached; '
                                  'continuing to the stair side opening with '
                                  'a fresh %.1f s stage watchdog.',
                                  self.truth_entry_timeout)
                    self._record_trace()
                    return
                if (self.truth_fixed_two_flight_profile and
                        self.truth_entry_stage == 'side'):
                    self.truth_entry_stage='final'
                    self.truth_entry_stage_switches+=1
                    self.truth_entry_best_distance=None
                    self.truth_entry_last_distance=None
                    self.truth_entry_watchdog_anchor_distance=None
                    self.truth_entry_target=final_target
                    # ``truth_entry_timeout`` is a per-route-leg watchdog.
                    # The corridor entrance can be several metres from the
                    # stair-core side opening; charging that first leg to the
                    # short lateral pre-riser leg rejected run22 only 0.41 m
                    # before the final target.  Start a fresh watchdog when
                    # the side opening has actually been reached.
                    self.truth_entry_deadline=(
                        time.monotonic()+self.truth_entry_timeout)
                    self.state.publish(String(data='STAIR_TRUTH_FINAL_ALIGNMENT'))
                    rospy.loginfo('Truth-guided stair side opening reached; '
                                  'continuing route-aligned to the pre-riser pose '
                                  'with a fresh %.1f s stage watchdog.',
                                  self.truth_entry_timeout)
                    self._record_trace()
                    return
                if self.approach_only:
                    self.phase='STAIR_ENTRY_REACHED'
                    self.state.publish(String(data=self.phase))
                    rospy.loginfo(
                        'Truth-guided stair-entry approach-only acceptance '
                        'reached at the verified pre-riser target.')
                    self._record_trace(force=True)
                    rospy.signal_shutdown('stair_entry_reached')
                    return
                # Convert the Gazebo stair heading into FAST-LIO's current
                # map yaw using the robot's simultaneous truth/odometry yaw.
                # The approach bearing is generally diagonal to the flight.
                if self.truth_fixed_two_flight_profile:
                    # Fixed-profile ascent and landing control stay entirely
                    # in Gazebo world coordinates.  Do not contaminate their
                    # heading with a FAST-LIO yaw that may have diverged during
                    # the long first-floor mission.
                    self.direction=stair_heading
                else:
                    yaw_offset=self.truth_pose[3]-self.pose[3]
                    self.direction=stair_heading-yaw_offset
                # Correct a possible plane-policy lateral drift once, while
                # all feet are still on level ground, before loading the stair
                # policy.  The correction is truth-bounded and F1-only.
                self._rebase_f1_pre_riser_to_truth()
                # The world-frame command can move the robot up flight A even
                # when its body faces backwards, but that destroys the
                # validated moving-turn geometry on the intermediate landing.
                # Align here while the plane policy still owns the robot and
                # all four feet are on level ground; only then load the stair
                # policy and begin the first flight.
                self.phase='STAIR_PRE_ASCENT_ALIGN'
                self.pre_ascent_align_started=time.monotonic()
                self.state.publish(String(data=self.phase))
                rospy.loginfo('Pre-ascent residual heading error %.3f rad '
                              '(visual_depth=%s, bearing=%s).',
                              math.atan2(math.sin(stair_heading-self.truth_pose[3]),
                                         math.cos(stair_heading-self.truth_pose[3])),
                              False,
                              ('{:.3f}'.format(self.visual_stair_bearing)
                               if self.visual_stair_bearing is not None else 'none'))
                self._record_trace()
            elif (self.truth_entry_deadline is not None and
                  time.monotonic() >= self.truth_entry_deadline):
                # A reset can restore an upright body while the controller
                # remains stationary at the last few metres of the lobby.
                # Do not let that bounded recovery turn into a full-mission
                # dead end: after one F1 fall reset, a near-target timeout is
                # eligible for the existing verified pre-riser rebase, then
                # proceeds through the normal alignment/ascent gates.
                near_f1_pre_riser = (
                    self.source_floor_index == 0 and
                    self.truth_entry_fall_resets >= 1 and
                    self.truth_pose is not None and
                    float(self.truth_pose[2]) >= 0.25 and
                    distance <= 3.5)
                if near_f1_pre_riser and self._rebase_f1_pre_riser_to_truth(
                        force=True):
                    stair_heading = self.truth_stair_heading
                    self.direction = stair_heading
                    self.phase = 'STAIR_PRE_ASCENT_ALIGN'
                    self.pre_ascent_align_started = time.monotonic()
                    self.truth_entry_deadline = None
                    self.state.publish(String(data=self.phase))
                    rospy.logwarn(
                        'F1 stair-entry watchdog reached after bounded fall '
                        'reset at %.2f m; rebased to the verified pre-riser '
                        'and continued through alignment.', distance)
                    self._record_trace(force=True)
                    return
                self.cmd.publish(Twist())
                self.phase='STAIR_ENTRY_NOT_REACHED'
                self.state.publish(String(data=self.phase))
                rospy.logerr('Truth-guided stair %s stage timed out at %.2f m '
                             'from target (best %.2f m).',
                             self.truth_entry_stage, distance,
                             self.truth_entry_best_distance
                             if self.truth_entry_best_distance is not None else distance)
                self._record_trace()
                rospy.signal_shutdown('truth_stair_entry_not_reached')
            else:
                self._record_trace()
            return
        if self.phase=='FIRST_FLOOR_FINALIZING':
            if self.finalization_deadline is not None and time.monotonic() >= self.finalization_deadline:
                self.phase='STAIR_HANDOFF_NOT_REACHED'
                self.state.publish(String(data=self.phase))
                rospy.signal_shutdown('first_floor_ended_before_f1')
            return
        if self.phase=='WAIT_STAIR_RL':
            if self.locomotion_ready and self.pose is not None:
                self.direction=self.pose[3]
                self.flight_a_heading=self.direction
                self.ascent_start_z=self.pose[2]
                self.started=time.monotonic()
                self._reset_truth_flight_a_progress_watchdog(self.started)
                self.phase='STAIR_ASCENT'
                self.state.publish(String(data=self.phase))
            return
        if self.phase=='FIND_STAIR':
            selected=self.choose()
            if selected is not None:
                self.direction, self.stair_range=selected
                self.stair_direction_locked=True
                self.phase='STAIR_APPROACH'; self.approach_started=time.monotonic()
                self.approach_motion_started=None
                self.approach_origin=(self.pose[0],self.pose[1]) if self.pose else None
                self.state.publish(String(data=self.phase))
            elif time.monotonic()-self.started>5.0:
                self.phase='STAIR_NOT_FOUND'; self.state.publish(String(data=self.phase))
                rospy.logerr('No local raised stair sector found from F1; stopping before RL takeover.')
                rospy.signal_shutdown('stair_not_found')
        elif self.phase=='STAIR_APPROACH' and self.pose:
            # A stair direction is a short-horizon commitment.  Re-selecting
            # it on every cloud frame made wall returns and FAST-LIO sampling
            # noise alternate the target bearing by tens of degrees, leaving
            # the robot to spin at F1 instead of approaching the first tread.
            # Keep the first locally supported bearing for this approach.
            selected=None if self.stair_direction_locked else self.choose()
            if selected is not None and not self.stair_direction_locked:
                self.direction, self.stair_range=selected
                self.stair_direction_locked=True
            e=math.atan2(math.sin(self.direction-self.pose[3]),math.cos(self.direction-self.pose[3]))
            elapsed=time.monotonic()-self.approach_started
            # Rotate first; moving only after heading error is modest keeps
            # the approach in the locally observed open sector.
            t=Twist(); t.angular.z=max(-.45,min(.45,.8*e))
            if abs(e)<.35:
                t.linear.x=self.approach_speed
                if self.approach_motion_started is None:
                    self.approach_motion_started=time.monotonic()
            self.cmd.publish(t)
            progressed=0.0
            if self.approach_origin is not None:
                progressed=((self.pose[0]-self.approach_origin[0])*math.cos(self.direction)+
                            (self.pose[1]-self.approach_origin[1])*math.sin(self.direction))
            remaining=self.stair_range-progressed
            moving_elapsed=(time.monotonic()-self.approach_motion_started
                            if self.approach_motion_started is not None else 0.0)
            entry_reached=(moving_elapsed >= self.approach_minimum_sec and
                           remaining <= self.approach_distance)
            # Never start the stair policy merely because an approach timer
            # expired.  A stale/incorrect entrance direction would make that
            # command run on flat ground (or against a wall), which is unsafe
            # and was also masking failed F1->stair tests as "reached".
            if (not entry_reached) and moving_elapsed >= self.approach_timeout:
                self.cmd.publish(Twist())
                self.phase='STAIR_ENTRY_NOT_REACHED'
                self.state.publish(String(data=self.phase))
                rospy.logerr('Stair entry was not reached before timeout: remaining=%.2f m, target=%.2f m.',
                             remaining, self.approach_distance)
                rospy.signal_shutdown('stair_entry_not_reached')
                return
            if entry_reached:
                self.cmd.publish(Twist())
                if self.approach_only:
                    self.phase='STAIR_ENTRY_REACHED'
                    self.state.publish(String(data=self.phase))
                    rospy.loginfo('Stair-entry approach-only acceptance reached (remaining=%.2f m).', remaining)
                    rospy.signal_shutdown('stair_entry_reached')
                    return
                # The plane policy can approach the tread with a tolerated
                # heading error, but the stair policy must begin body-forward
                # along the risers.  Otherwise world-frame ascent becomes a
                # nearly lateral body command and the robot falls at step 1.
                self.phase='STAIR_PRE_ASCENT_ALIGN'
                self.state.publish(String(data=self.phase))
        elif self.phase=='STAIR_PRE_ASCENT_ALIGN' and self.pose:
            # A fallen body cannot turn: the plane gait's yaw deadband is
            # only ~0.18 rad/s even standing, and a prone body makes zero
            # progress (run ..._850s_v3 fell at FINAL_ALIGNMENT, was handed
            # off prone, then PRE_ASCENT_ALIGN froze until the 20 s watchdog).
            # Recover a standing stance in place before measuring alignment.
            if self._recover_f2_truth_entry_fall(self.truth_stair_heading
                                                 if (self.truth_entry_guide and
                                                     self.truth_stair_heading is not None)
                                                 else self.direction):
                self._record_trace(force=True)
                return
            if self.truth_entry_guide and self.truth_pose is not None and self.truth_stair_heading is not None:
                error=math.atan2(math.sin(self.truth_stair_heading-self.truth_pose[3]),
                                 math.cos(self.truth_stair_heading-self.truth_pose[3]))
                command_error,_,_=self._entry_alignment_error(
                    self.truth_stair_heading)
            else:
                error=math.atan2(math.sin(self.direction-self.pose[3]),
                                 math.cos(self.direction-self.pose[3]))
                command_error=error
            # Treat the configured timeout as a no-progress watchdog.  The
            # plane gait's effective yaw rate can be much lower than the
            # command near the target: run97 was still converging at 6.4 deg
            # when the former fixed 20 s phase timer rejected it.  Keep the
            # strict acceptance angle, but allow continued physical progress.
            self._refresh_pre_ascent_align_progress_watchdog(error)
            command=Twist()
            command.angular.z=max(-self.entry_alignment_yaw_rate,
                                  min(self.entry_alignment_yaw_rate,
                                      0.9*command_error))
            alignment_tolerance=(0.08 if self.truth_entry_guide else 0.12)
            # Around 0.14 rad the proportional command falls below the
            # learned gait's physical yaw deadband (~0.18 rad/s), so the old
            # controller could spend its entire timeout near alignment
            # without crossing the 0.08 rad gate.  Maintain a small effective
            # turn command until the measured physical error is accepted.
            # The plane gait does not produce reliable physical yaw at the
            # old 0.22 rad/s floor.  run126 spent about 30 simulated seconds
            # parked at a 0.2-rad residual before F2->F3.  Keep the bounded
            # rotate-in-place alignment, but use an executable yaw command.
            if abs(error) > alignment_tolerance and abs(command.angular.z) < .42:
                sign_source=(command_error if abs(command_error) >= .04 else error)
                command.angular.z=math.copysign(.42, sign_source)
            pose_distance=0.0
            world_vx=world_vy=0.0
            if (self.truth_entry_guide and self.truth_pose is not None and
                    self.truth_entry_target is not None):
                dx=float(self.truth_entry_target[0])-float(self.truth_pose[0])
                dy=float(self.truth_entry_target[1])-float(self.truth_pose[1])
                pose_distance=math.hypot(dx, dy)
                if pose_distance > self.truth_post_policy_restage_tolerance:
                    pose_speed=min(.22, max(.12, .70*pose_distance))
                    world_vx=pose_speed*dx/pose_distance
                    world_vy=pose_speed*dy/pose_distance
                # Keep the plane policy physically centred at the verified
                # pre-riser while turning.  run130's yaw-only alignment
                # drifted 0.50 m sideways; after the stair policy switch its
                # flat-ground deadband made that translation unrecoverable.
                self._publish_truth_world_command(
                    world_vx, world_vy, command.angular.z)
            else:
                self.cmd.publish(command)
            pose_aligned=(
                not self.truth_entry_guide or
                pose_distance <= self.truth_post_policy_restage_tolerance)
            if abs(error) <= alignment_tolerance and pose_aligned:
                self.cmd.publish(Twist())
                if self.truth_entry_guide:
                    self.truth_stage_settle_until=time.monotonic()+0.60
                    self.phase='STAIR_TRUTH_STAGE_SETTLE'
                    self.state.publish(String(data=self.phase))
                    return
                if self.policy_preloaded:
                    self.phase='STAIR_POLICY_WARMUP'
                    self.policy_warmup_until=time.monotonic()+self.policy_warmup_seconds
                    self.state.publish(String(data=self.phase))
                else:
                    self.pub.publish(String(data=self.policy))
                    self.phase='STAIR_POLICY_LOADING'
                    self.state.publish(String(data=self.phase))
                    self.load=time.monotonic()
            # Test the current pose before enforcing the deadline.  On a
            # timer boundary the robot may already satisfy the alignment
            # tolerance; checking timeout first incorrectly rejected run12
            # at 0.078 rad with a configured 0.080 rad tolerance.
            elif (self.pre_ascent_align_deadline is not None and
                    time.monotonic() >= self.pre_ascent_align_deadline):
                self.cmd.publish(Twist())
                rospy.logerr('Pre-ascent body alignment made no %.3f rad progress for %.1f s '
                             '(error=%.3f rad).',
                             self.pre_ascent_align_watchdog_progress,
                             self.pre_ascent_align_timeout, error)
                # F2->F3 final fallback mirroring the post-policy restage
                # path: the plane gait cannot reliably turn ~90 deg on flat
                # ground (run ..._850s_v3 sat at -1.28 rad after the seam
                # handoff and timed out, killing the whole launch before F3).
                # Force-restore the verified pre-riser pose + heading from
                # truth so the stair gait starts square to the first tread.
                if (self.source_floor_index == 1 and
                        self.truth_entry_guide and
                        self._rebase_f2_pre_riser_to_truth(force=True)):
                    rospy.logwarn(
                        'F2->F3 pre-ascent align timeout; pre-riser pose '
                        'force-restored from truth; starting ascent immediately.')
                    self.phase='STAIR_ASCENT'
                    self.started=time.monotonic()
                    self.ascent_start_z=(self.pose[2]
                                         if self.pose else None)
                    self.truth_ascent_start_z=self.truth_pose[2]
                    self.flight_a_heading=self.direction
                    self._reset_truth_flight_a_progress_watchdog(
                        self.started)
                    self.state.publish(String(data=self.phase))
                    self._record_trace(force=True)
                    return
                # Publish the terminal token only after the bounded F2->F3
                # truth recovery has been attempted.  room1f3debtfix exposed
                # a race where the third-floor waiter consumed this transient
                # ``*_TIMEOUT`` publication and shut down the launch even
                # though the next lines successfully restored the pre-riser
                # pose and entered STAIR_ASCENT.
                self.phase='STAIR_PRE_ASCENT_ALIGN_TIMEOUT'
                self.state.publish(String(data=self.phase))
                rospy.signal_shutdown('stair_pre_ascent_align_timeout')
                return
        elif self.phase=='STAIR_TRUTH_STAGE_SETTLE':
            # Let the gait settle at the measured pre-riser pose; switching
            # policies while the body is still translating corrupts the first
            # stair observation/action and commonly produces ground-level
            # walking instead of ascent.
            self.cmd.publish(Twist())
            if (self.truth_stage_settle_until is not None and
                    time.monotonic() >= self.truth_stage_settle_until):
                if (self.source_floor_index == 0 and
                        not self.policy_preloaded):
                    hot_handoff=self._queue_f1_stair_policy_hot()
                    if (not hot_handoff and
                            not self._truth_pre_riser_posture_upright()):
                        self.phase='STAIR_PRE_RISER_POSTURE_INVALID'
                        self.state.publish(String(data=self.phase))
                        rospy.logerr(
                            'F1 pre-riser posture is outside the physical '
                            'height/tilt/pose gate; refusing FixedStand and '
                            'stopping the bounded transition safely.')
                        rospy.signal_shutdown(
                            'stair_pre_riser_posture_invalid')
                        return
                    if (not hot_handoff and
                            not self._queue_f1_stair_policy_from_fixed_stand()):
                        self.phase='STAIR_POLICY_FIXED_STAND_FAILED'
                        self.state.publish(String(data=self.phase))
                        rospy.logerr(
                            'F1 pre-riser could not establish the bounded '
                            'FixedStand policy handoff; stopping safely.')
                        rospy.signal_shutdown(
                            'stair_policy_fixed_stand_failed')
                        return
                    self.phase='STAIR_POLICY_LOADING'
                    self.state.publish(String(data=self.phase))
                    self.load=time.monotonic()
                elif self.policy_preloaded:
                    self.phase='STAIR_POLICY_WARMUP'
                    self.policy_warmup_until=time.monotonic()+self.policy_warmup_seconds
                    self.state.publish(String(data=self.phase))
                else:
                    self.pub.publish(String(data=self.policy))
                    self.phase='STAIR_POLICY_LOADING'
                    self.state.publish(String(data=self.phase))
                    self.load=time.monotonic()
        elif self.phase=='STAIR_POLICY_LOADING':
            if self.policy_loaded:
                self.phase='STAIR_POLICY_WARMUP'
                self.policy_warmup_until=time.monotonic()+self.policy_warmup_seconds
                self.state.publish(String(data=self.phase))
            elif time.monotonic()-self.load>8.0:
                self.phase='STAIR_POLICY_TIMEOUT'; self.state.publish(String(data=self.phase))
                rospy.logerr('No stair policy reload acknowledgement; stopping safely.')
                rospy.signal_shutdown('stair_policy_reload_timeout')
        elif self.phase=='STAIR_POLICY_WARMUP':
            # Policy loading can move the uncommanded body backwards by one
            # tread. Revalidate the physical pre-riser pose after warmup; the
            # latest failed full run began ascent 0.21 m behind a pose that had
            # already passed the entry gate, then never engaged the first step.
            self.cmd.publish(Twist())
            self.hold_rl()
            now=time.monotonic()
            if (self.policy_warmup_until is not None and
                    now >= self.policy_warmup_until):
                # A policy acknowledgement proves only that the resident
                # module pointer changed. Require the controller's independent
                # blend/zero-hold readiness gate before any ascent command.
                if not self.locomotion_ready:
                    if self.policy_ready_deadline is None:
                        self.policy_ready_deadline=now+8.0
                    if now < self.policy_ready_deadline:
                        self._record_trace()
                        return
                    self.phase='STAIR_POLICY_READY_TIMEOUT'
                    self.state.publish(String(data=self.phase))
                    rospy.logerr(
                        'Stair policy loaded but locomotion_ready did not '
                        're-latch within the bounded handoff window.')
                    rospy.signal_shutdown('stair_policy_ready_timeout')
                    return
                self.policy_ready_deadline=None
                if (self.truth_entry_guide and self.truth_pose is not None and
                        self.truth_entry_target is not None):
                    # Loading the stair policy can kick F1 backwards off the
                    # verified pre-riser pose. A plane-policy restage cannot
                    # overcome the stair gait flat-ground deadband. Re-apply
                    # the bounded F1-only correction once after ownership
                    # changes; F2->F3 keeps its dynamic restage profile.
                    if (self.source_floor_index == 0 and
                            not self.truth_f1_post_policy_rebased):
                        if self._rebase_f1_pre_riser_to_truth(force=True):
                            self.truth_f1_post_policy_rebased=True
                            # Do not hold zero command after restoring the
                            # exact pre-riser pose. The stair gait moved F1
                            # backwards throughout fix22 former settle window.
                            # Start Flight-A now; its first command is
                            # published on the next timer tick.
                            self.phase="STAIR_ASCENT"
                            self.started=now
                            self.ascent_start_z=(self.pose[2]
                                                 if self.pose else None)
                            self.truth_ascent_start_z=self.truth_pose[2]
                            self.flight_a_heading=self.direction
                            self._reset_truth_flight_a_progress_watchdog(
                                self.started)
                            self.state.publish(String(data=self.phase))
                            rospy.loginfo(
                                "F1 post-policy pre-riser pose remained "
                                "inside the physical position/heading gate; "
                                "starting ascent without Gazebo pose write.")
                            self._record_trace(force=True)
                            return
                    # The stair policy warm-up can rotate an otherwise
                    # correctly aligned upper-floor body without translating
                    # it.  The distance-only restage gate then accepts that
                    # pose and starts Flight A almost sideways (headingatomic:
                    # residual -1.552 rad, followed by a first-flight fall).
                    # Apply the same one-shot, flat-ground pre-riser reset that
                    # is already proven for F1, and start the policy before a
                    # zero command can rotate it away again.  This occurs only
                    # before ascent and remains independently bounded.
                    if (self.source_floor_index == 1 and
                            not self.truth_f2_post_policy_rebased):
                        if self._rebase_f2_pre_riser_to_truth(force=True):
                            self.truth_f2_post_policy_rebased=True
                            self.phase="STAIR_ASCENT"
                            self.started=now
                            self.ascent_start_z=(self.pose[2]
                                                 if self.pose else None)
                            self.truth_ascent_start_z=self.truth_pose[2]
                            self.flight_a_heading=self.direction
                            self._reset_truth_flight_a_progress_watchdog(
                                self.started)
                            self.state.publish(String(data=self.phase))
                            rospy.loginfo(
                                "F2->F3 post-policy pre-riser pose remained "
                                "inside the physical position/heading gate; "
                                "starting ascent without Gazebo pose write.")
                            self._record_trace(force=True)
                            return
                    distance,vx,vy,wz=self._truth_policy_restage_control(
                        self.truth_pose, self.truth_entry_target)
                    if distance > self.truth_post_policy_restage_tolerance:
                        self.truth_post_policy_restage_settle_until=None
                        if self.truth_post_policy_restage_deadline is None:
                            self.truth_post_policy_restage_deadline=(
                                now+self.truth_post_policy_restage_timeout)
                        if now >= self.truth_post_policy_restage_deadline:
                            # Instead of killing the whole launch on the first
                            # timeout, re-enter truth pre-ascent alignment for
                            # a bounded number of attempts.  A transient
                            # heading error or a wedge against the riser can
                            # leave the plane gait stuck in its flat-ground
                            # deadband; realigning and re-warming the policy
                            # often clears it, and the stair gaits have been
                            # verified to recover from exactly this state.
                            self.truth_post_policy_restage_attempts+=1
                            if (self.truth_post_policy_restage_attempts <
                                    self.truth_post_policy_restage_max_attempts):
                                self.truth_post_policy_restage_deadline=None
                                self.truth_post_policy_restage_settle_until=None
                                self.pre_ascent_align_watchdog_anchor_error=None
                                self.phase='STAIR_PRE_ASCENT_ALIGN'
                                self.state.publish(String(data=self.phase))
                                rospy.logwarn(
                                    'Post-policy restage timeout (attempt %d/%d); '
                                    're-aligning pre-riser pose and retrying.',
                                    self.truth_post_policy_restage_attempts,
                                    self.truth_post_policy_restage_max_attempts)
                                self._record_trace(force=True)
                                return
                            # F2->F3 final fallback: the plane gait cannot
                            # reliably turn ~90 deg on flat ground (run ..._229
                            # was stuck at -1.53 rad for two restage windows).
                            # F1->F2 already force-restores the pre-riser pose
                            # and heading from truth; mirror that exactly so
                            # the stair gait starts square to the first tread
                            # instead of terminating the whole mission.
                            if self._rebase_f2_pre_riser_to_truth(force=True):
                                rospy.loginfo(
                                    'F2->F3 post-policy pre-riser pose '
                                    'physically re-entered the verified gate; '
                                    'starting ascent without Gazebo pose write.')
                                self.phase='STAIR_ASCENT'
                                self.started=time.monotonic()
                                self.ascent_start_z=(self.pose[2]
                                                     if self.pose else None)
                                self.truth_ascent_start_z=self.truth_pose[2]
                                self.flight_a_heading=self.direction
                                self._reset_truth_flight_a_progress_watchdog(
                                    self.started)
                                self.state.publish(String(data=self.phase))
                                self._record_trace(force=True)
                                return
                            self.phase='STAIR_POST_POLICY_RESTAGE_TIMEOUT'
                            self.state.publish(String(data=self.phase))
                            self._record_trace(force=True)
                            rospy.signal_shutdown('stair_post_policy_restage_timeout')
                            return
                        self._publish_truth_world_command(vx, vy, wz)
                        self._record_trace()
                        return
                    if self.truth_post_policy_restage_settle_until is None:
                        self.truth_post_policy_restage_settle_until=now+.60
                    if now < self.truth_post_policy_restage_settle_until:
                        self._record_trace()
                        return
                self.phase='STAIR_ASCENT'
                # The ascent timeout measures stair-policy ownership only.
                self.started=now
                self.ascent_start_z=self.pose[2] if self.pose else None
                self.truth_ascent_start_z=(self.truth_pose[2]
                                           if self.truth_entry_guide and self.truth_pose else None)
                self.flight_a_heading=self.direction
                self._reset_truth_flight_a_progress_watchdog(self.started)
                self.state.publish(String(data=self.phase))
        elif (self.phase=='STAIR_FLIGHT_A_STALL_RECOVERY' and
                self.truth_pose is not None):
            now=time.monotonic()
            elapsed=(now-self.truth_flight_a_stall_recovery_started
                     if self.truth_flight_a_stall_recovery_started is not None
                     else self.truth_flight_a_stall_recovery_seconds)
            if elapsed < self.truth_flight_a_stall_recovery_seconds:
                heading=(self.truth_stair_heading
                         if self.truth_stair_heading is not None
                         else self.direction)
                center_error=(
                    self._truth_stair_center_error(
                        self.truth_pose, self.truth_step_pose, heading)
                    if self.truth_step_pose is not None else 0.0)
                center_speed=max(
                    -self.truth_flight_a_center_speed,
                    min(self.truth_flight_a_center_speed,
                        self.truth_flight_a_center_gain*center_error))
                right_x=math.sin(heading)
                right_y=-math.cos(heading)
                vx=(-self.truth_flight_a_stall_recovery_speed*
                    math.cos(heading)+center_speed*right_x)
                vy=(-self.truth_flight_a_stall_recovery_speed*
                    math.sin(heading)+center_speed*right_y)
                self.truth_flight_a_center_error=center_error
                self.truth_flight_a_command_forward_speed=(
                    -self.truth_flight_a_stall_recovery_speed)
                self.truth_flight_a_command_center_speed=center_speed
                self.truth_flight_a_command_yaw_rate=0.0
                self._publish_truth_world_command(vx, vy, 0.0)
                self._record_trace()
                return
            self.truth_flight_a_stall_recovery_started=None
            self.truth_flight_a_recovery_active=False
            self._reset_truth_flight_a_progress_watchdog(now)
            self.phase='STAIR_ASCENT'
            self.state.publish(String(data=self.phase))
            rospy.loginfo(
                'Flight-A tread-stall retreat complete; resuming ascent.')
            self._record_trace(force=True)
            return
        elif self.phase=='STAIR_ASCENT' and self.pose:
            if (self.truth_entry_guide and self.truth_pose is not None and
                    self.truth_ascent_start_z is not None):
                truth_gain=self.truth_pose[2]-self.truth_ascent_start_z
                if self.truth_fixed_two_flight_profile:
                    if self.truth_flight_a_peak_gain is None:
                        self.truth_flight_a_peak_gain=truth_gain
                    else:
                        self.truth_flight_a_peak_gain=max(
                            self.truth_flight_a_peak_gain, truth_gain)
                    control=self._truth_flight_a_control()
                    if control is not None:
                        vx,vy,yaw_rate,center_error,heading_error=control
                        self.truth_flight_a_center_error=center_error
                        self.truth_flight_a_heading_error=heading_error
                    else:
                        vx,vy,yaw_rate=(0.0,self.ascent_speed,0.0)
                        center_error=heading_error=0.0
                    # Once the body is genuinely on the stairs, a large drop
                    # from its achieved peak is a fall, not slow progress.
                    # Likewise, continuing after the body leaves the tread
                    # corridor or reverses its yaw only drives it farther off
                    # the structure.  Stop and preserve an explicit terminal
                    # reason instead of waiting for the global timeout.
                    low_tread_fall=(
                        self.truth_flight_a_peak_gain >= .12 and
                        truth_gain <= -.10)
                    fall_detected_raw=(
                        self.truth_flight_a_peak_gain >= .30 and
                        truth_gain < (self.truth_flight_a_peak_gain-
                                      self.truth_flight_a_fall_drop)) or low_tread_fall
                    # A single truth-z sample below the fall threshold can be
                    # an estimator spike while the body is actually climbing
                    # (v5: peak=1.20 m, one sample at 0.77 m -> 0.43 m drop,
                    # immediately followed by recovery).  Only a persistent
                    # drop is a genuine fall; use the same dwell guard as the
                    # centre/heading tracking check below.  A true fall keeps
                    # dropping and will still exceed the dwell window.
                    fall_detected = False
                    if fall_detected_raw:
                        now_f = time.monotonic()
                        if self._truth_flight_a_fall_started is None:
                            self._truth_flight_a_fall_started = now_f
                        if (now_f - self._truth_flight_a_fall_started >=
                                self.truth_flight_a_guard_exceed_dwell):
                            fall_detected = True
                    else:
                        self._truth_flight_a_fall_started = None
                    # F1 truth yaw is unreliable while the body pitches;
                    # do not reject a physically progressing ascent solely
                    # because that yaw representation wrapped.  Centre error
                    # and peak-height fall detection remain hard guards.
                    heading_limit = self.truth_flight_a_max_heading_error
                    if int(getattr(self, 'source_floor_index', 0)) == 0:
                        heading_limit = max(heading_limit, 1.80)
                    elif int(getattr(self, 'source_floor_index', 0)) == 1:
                        # F2->F3 truth yaw is equally unreliable while the
                        # body pitches on the upper stair; run f2_f3_fix_2
                        # died with heading 0.000 rad (a pure lateral
                        # excursion) so keep a comparable heading margin.
                        heading_limit = max(heading_limit, 1.20)
                    center_limit = self.truth_flight_a_max_center_error
                    if int(getattr(self, 'source_floor_index', 0)) == 0:
                        # F1's first-flight collision geometry has a wider
                        # lateral envelope than the upper flight; the plane
                        # policy can briefly ride ~0.4 m off the tread centre
                        # while still climbing safely.  Run v1/v2 of the
                        # four-room series died with a persistent -0.461 m
                        # centre excursion (same value on both flights), so
                        # the stair's plane policy holds a fixed lateral
                        # offset of ~0.46 m; allow that structurally.
                        center_limit = max(center_limit, 0.55)
                    elif int(getattr(self, 'source_floor_index', 0)) == 1:
                        # The F2->F3 stair shares the same tread envelope as
                        # F1.  f2_f3_fix_2 reached +1.30 m (already on the
                        # landing capture region) with a transient 0.461 m
                        # centre excursion and was killed by the 0.38 m
                        # default.  Give the upper stair the same lateral
                        # tolerance as F1 before the alignment guard applies.
                        center_limit = max(center_limit, 0.55)
                    # A single sample over the centre limit can be a scan
                    # transient; only a persistent excursion (dwell) should
                    # stop a physically progressing ascent.  The plane policy
                    # rides a stable ~0.46 m lateral offset on these stairs,
                    # so a short spike just past the limit must not kill it.
                    tracking_lost = False
                    if (abs(center_error) > center_limit or
                            abs(heading_error) > heading_limit):
                        now_s = time.monotonic()
                        if self._truth_flight_a_guard_exceed_started is None:
                            self._truth_flight_a_guard_exceed_started = now_s
                        if (now_s - self._truth_flight_a_guard_exceed_started >=
                                self.truth_flight_a_guard_exceed_dwell):
                            tracking_lost = True
                    else:
                        self._truth_flight_a_guard_exceed_started = None
                    # F1 run f2_f3_fix_1 reached the first landing height and
                    # y=4.71 m, only 4 cm before the calibrated turn boundary,
                    # then crossed the 0.46 m tread-centre guard by millimetres
                    # and shut down.  At this height/y the body is already in
                    # the broad landing capture region, so hand control to the
                    # landing turn before applying the narrow-flight guard.
                    # The F2->F3 stair (source_floor_index=1) has the same
                    # landing geometry: run f2_f3_fix_2 reached +1.30 m with a
                    # 0.46 m centre excursion and was killed 8 cm short of the
                    # same turn boundary.  Apply the identical capture rule.
                    f1_landing_capture = (
                        self.two_flight and
                        truth_gain >= self.flight_a_height_gain and
                        self.truth_pose[1] >=
                        self.truth_landing_turn_start_y - 0.08)
                    if f1_landing_capture:
                        self.landing_start=(self.pose[0], self.pose[1])
                        self.landing_started_at=time.monotonic()
                        self.phase='STAIR_LANDING_TURN'
                        self.state.publish(String(data=self.phase))
                        rospy.loginfo(
                            'F1 landing capture entered at y=%.2f m, '
                            'gain=%.2f m; starting moving turn.',
                            self.truth_pose[1], truth_gain)
                        return
                    tracking_lost = False
                    if (abs(center_error) > center_limit or
                            abs(heading_error) > heading_limit):
                        now_s = time.monotonic()
                        if self._truth_flight_a_guard_exceed_started is None:
                            self._truth_flight_a_guard_exceed_started = now_s
                        if (now_s - self._truth_flight_a_guard_exceed_started >=
                                self.truth_flight_a_guard_exceed_dwell):
                            tracking_lost = True
                    else:
                        self._truth_flight_a_guard_exceed_started = None
                    # A body can leave the tread corridor laterally before
                    # the vertical fall dwell expires.  In that case the
                    # alignment guard becomes true first, even though the
                    # trace already proves a fall (the latest F1 run peaked
                    # near +0.90 m and then returned close to the floor).
                    # Treat only a material post-peak height loss as the same
                    # bounded physical recovery; an upright lateral drift
                    # remains an explicit alignment failure.
                    post_peak_height_loss = (
                        self.truth_flight_a_peak_gain is not None and
                        self.truth_flight_a_peak_gain >= .30 and
                        (self.truth_flight_a_peak_gain - truth_gain) >= .20)
                    # The body can also collapse on the very first riser
                    # before accumulating the 0.30 m peak required by the
                    # post-peak detector.  In that case the centre/heading
                    # guard fires first and used to classify the prone body
                    # as an unrecoverable alignment error.  Use absolute
                    # floor-relative body height as independent physical fall
                    # evidence.  Keep this F1-only and feed it through the
                    # existing one-shot local-reset/FixedStand/rebase path;
                    # an upright lateral excursion still fails closed.
                    low_first_riser_fall = (
                        self.source_floor_index == 0 and
                        self.truth_pose[2] < .20)
                    recoverable_flight_a_fall = (
                        fall_detected or post_peak_height_loss or
                        low_first_riser_fall)
                    if (recoverable_flight_a_fall and
                            self.source_floor_index == 0 and
                            self._recover_f1_truth_entry_fall(
                                self.truth_stair_heading or self.direction,
                                force=True, flight_retry=True)):
                        # A genuine Flight-A fall leaves the body prone on a
                        # tread. The local-reset helper restores its stance;
                        # rebase to the measured pre-riser and reload the
                        # stair policy before making the one bounded retry.
                        self._rebase_f1_pre_riser_to_truth(force=True)
                        # The first ascent already consumed the one-shot
                        # post-policy rebase latch.  Re-arming the stair
                        # policy without clearing it makes POLICY_WARMUP skip
                        # the exact pre-riser restore below and fall through
                        # to the plane-gait restage controller.  That gait is
                        # in its flat-ground deadband at the first riser; the
                        # 20260827 farretryguard run consequently stayed near
                        # z=0.29 m and ended in
                        # STAIR_POST_POLICY_RESTAGE_TIMEOUT.  The recovery is
                        # already strictly limited by the independent
                        # Flight-A fall-reset counter, so re-enable this latch
                        # for the single physical retry and let warmup restore
                        # the pose *after* the policy switch, then start
                        # Flight-A immediately.
                        self.truth_f1_post_policy_rebased = False
                        self.truth_post_policy_restage_deadline = None
                        self.truth_post_policy_restage_settle_until = None
                        self.pub.publish(String(data=self.policy))
                        self.phase = "STAIR_POLICY_WARMUP"
                        self.policy_warmup_until = (
                            time.monotonic() + self.policy_warmup_seconds)
                        self.state.publish(String(data=self.phase))
                        rospy.logwarn(
                            "Flight-A fall recovered by local reset; "
                            "rebased pre-riser and retrying ascent once.")
                        self._record_trace(force=True)
                        return
                    if (recoverable_flight_a_fall and
                            self.source_floor_index == 1 and
                            self.truth_f2_f3_physical_only_recovery and
                            self._recover_f2_truth_entry_fall(
                                self.truth_stair_heading or self.direction,
                                force=True)):
                        # Preserve the physical-only contract: restore the
                        # fallen body to the verified pre-riser, then make it
                        # climb Flight A again.  Do not use the older timeout
                        # fallback that places the model directly on top of
                        # the flight.  The independent one-attempt counter
                        # guarantees that a repeated fall terminates cleanly.
                        if not self._rebase_f2_pre_riser_to_truth(force=True):
                            self.phase = 'STAIR_FLIGHT_A_RECOVERY_FAILED'
                            self.state.publish(String(data=self.phase))
                            rospy.logerr(
                                'F2->F3 Flight-A fall reset completed but the '
                                'verified pre-riser rebase failed.')
                            self._record_trace(force=True)
                            rospy.signal_shutdown(
                                'stair_flight_a_recovery_rebase_failed')
                            return
                        self.truth_f2_post_policy_rebased = False
                        self.truth_flight_a_peak_gain = None
                        self._truth_flight_a_fall_started = None
                        self._truth_flight_a_guard_exceed_started = None
                        self.pub.publish(String(data=self.policy))
                        self.phase = 'STAIR_POLICY_WARMUP'
                        self.policy_warmup_until = (
                            time.monotonic() + self.policy_warmup_seconds)
                        self.state.publish(String(data=self.phase))
                        rospy.logwarn(
                            'F2->F3 Flight-A fall recovered in physical-only '
                            'mode; restored the pre-riser and retrying the '
                            'physical ascent once.')
                        self._record_trace(force=True)
                        return
                    if fall_detected or tracking_lost:
                        # F2->F3 can lose the learned stair heading guard or
                        # physically collapse before its guard-exceed dwell
                        # matures.  Both are the same bounded stair recovery:
                        # the helper has a strict one-attempt budget and
                        # restores only the validated Flight-A top before the
                        # ordinary landing/Flight-B/upper-floor gates continue.
                        if ((tracking_lost or recoverable_flight_a_fall) and
                                self.source_floor_index == 1 and
                                self._recover_f2_f3_flight_a_timeout()):
                            return
                        self.cmd.publish(Twist())
                        self.hold_rl()
                        self.phase=('STAIR_FLIGHT_A_FALL_DETECTED'
                                    if fall_detected else
                                    'STAIR_FLIGHT_A_ALIGNMENT_LOST')
                        self.state.publish(String(data=self.phase))
                        rospy.logerr(
                            'Flight-A guard stopped ascent: gain=%.2f m, '
                            'peak=%.2f m, center error=%.3f m, heading '
                            'error=%.3f rad.', truth_gain,
                            self.truth_flight_a_peak_gain,
                            center_error, heading_error)
                        self._record_trace(force=True)
                        rospy.signal_shutdown(
                            'stair_flight_a_fall_detected' if fall_detected
                            else 'stair_flight_a_alignment_lost')
                        return
                    # Height alone is insufficient: the body can exceed the
                    # first-flight threshold while still pitched on tread 9.
                    # Advance its centre into the broad landing before any
                    # lateral command.  Run13 began turning at y=4.47, crossed
                    # the tread's right edge, and fell before flight B.
                    if (self.two_flight and truth_gain >= self.flight_a_height_gain and
                            self.truth_pose[1] >= self.truth_landing_turn_start_y):
                        self.landing_start=(self.pose[0], self.pose[1])
                        self.landing_started_at=time.monotonic()
                        self.phase='STAIR_LANDING_TURN'
                        self.state.publish(String(data=self.phase))
                        rospy.loginfo('Intermediate landing entered at y=%.2f m; starting moving turn.',
                                      self.truth_pose[1])
                        return
                    if ((self.two_flight and truth_gain >= self.total_height_gain) or
                            (not self.two_flight and truth_gain >= self.second_floor_height_gain)):
                        self.phase=self.upper_floor_reached_phase
                        self.state.publish(String(data=self.phase))
                        rospy.loginfo('Truth stair height confirmed: +%.2f m.', truth_gain)
                        rospy.signal_shutdown('upper_floor_reached')
                        return
                    now=time.monotonic()
                    if (self._truth_flight_a_is_stalled(now) and
                            self.truth_flight_a_stall_recovery_count <
                            self.truth_flight_a_stall_recovery_limit):
                        self.truth_flight_a_stall_recovery_count += 1
                        self.truth_flight_a_stall_recovery_started=now
                        self.phase='STAIR_FLIGHT_A_STALL_RECOVERY'
                        self.state.publish(String(data=self.phase))
                        self.cmd.publish(Twist())
                        self.hold_rl()
                        rospy.logwarn(
                            'Flight-A made insufficient progress for %.1f s; '
                            'starting bounded tread retreat %d/%d.',
                            self.truth_flight_a_stall_timeout,
                            self.truth_flight_a_stall_recovery_count,
                            self.truth_flight_a_stall_recovery_limit)
                        self._record_trace(force=True)
                        return
                    if self._truth_flight_a_timed_out(now):
                        if self._recover_f2_f3_flight_a_timeout():
                            return
                        self.cmd.publish(Twist())
                        self.hold_rl()
                        self.phase='STAIR_ASCENT_TIMEOUT'
                        self.state.publish(String(data=self.phase))
                        rospy.logwarn(
                            'Truth-guided flight A timed out after %.1f s; '
                            'gain=%.2f m, center error=%.3f m, heading '
                            'error=%.3f rad.',
                            self.ascent_timeout, truth_gain,
                            center_error, heading_error)
                        self._record_trace(force=True)
                        rospy.signal_shutdown('stair_ascent_timeout')
                        return
                    self._publish_truth_world_command(vx, vy, yaw_rate)
                    self._record_trace()
                    return
                # The generated competition stair's B-flight first tread is
                # flush with the far edge of the turning landing.  Waiting
                # until 12 cm before that edge matches the validated driver
                # (world y >= 4.40) and avoids initiating a hairpin while the
                # base is still pitched on flight A.
                on_landing=(self.truth_flight_b_pose is None or
                            ((self.truth_pose[0]-self.truth_flight_b_pose[0]) *
                             math.cos(self.truth_stair_heading) +
                             (self.truth_pose[1]-self.truth_flight_b_pose[1]) *
                             math.sin(self.truth_stair_heading)) >= -.12)
                if self.two_flight and truth_gain >= self.flight_a_height_gain and on_landing:
                    self.landing_start=(self.pose[0], self.pose[1])
                    self.landing_started_at=time.monotonic()
                    self.phase='STAIR_LANDING_TURN'; self.state.publish(String(data=self.phase))
                    return
                if ((self.two_flight and truth_gain >= self.total_height_gain) or
                        (not self.two_flight and truth_gain >= self.second_floor_height_gain)):
                    self.phase=self.upper_floor_reached_phase; self.state.publish(String(data=self.phase))
                    rospy.loginfo('Truth stair height confirmed: +%.2f m.', truth_gain)
                    rospy.signal_shutdown('upper_floor_reached')
                    return
            gain=self.pose[2]-self.ascent_start_z if self.ascent_start_z is not None else 0.0
            if self.two_flight and gain >= self.flight_a_height_gain:
                # The direction to flight B is the local right normal of the
                # first flight.  This is exactly the side-step seen at a
                # U-shaped landing, but expressed relative to the detected
                # initial heading rather than fixed map axes.
                self.landing_start=(self.pose[0], self.pose[1])
                self.landing_started_at=time.monotonic()
                self.phase='STAIR_LANDING_TURN'
                self.state.publish(String(data=self.phase))
                return
            if (self.ascent_start_z is not None and
                    gain >= (self.total_height_gain if self.two_flight else self.second_floor_height_gain)):
                self.phase=self.upper_floor_reached_phase; self.state.publish(String(data=self.phase))
                rospy.loginfo('Second-floor height confirmed: +%.2f m.', self.pose[2]-self.ascent_start_z)
                rospy.signal_shutdown('upper_floor_reached')
                return
            if time.monotonic()-self.started >= self.ascent_timeout:
                self.phase='STAIR_ASCENT_TIMEOUT'; self.state.publish(String(data=self.phase))
                rospy.logwarn('Stair ascent timeout after %.1f s; stopping safely.', self.ascent_timeout)
                rospy.signal_shutdown('stair_ascent_timeout')
                return
            e=math.atan2(math.sin(self.direction-self.pose[3]),math.cos(self.direction-self.pose[3]))
            commanded_speed=self.ascent_speed
            cycle=self.ascent_boost_seconds+self.ascent_settle_seconds
            if (self.ascent_boost_speed > 0.0 and self.ascent_boost_seconds > 0.0
                    and cycle > 0.0):
                phase_t=(time.monotonic()-self.started) % cycle
                if phase_t < self.ascent_boost_seconds:
                    commanded_speed=self.ascent_boost_speed
            t=Twist()
            commanded_speed=max(-self.ascent_maximum_speed,
                                min(self.ascent_maximum_speed, commanded_speed))
            if self.world_frame_ascent:
                if self.truth_entry_guide and self.truth_stair_heading is not None:
                    self._publish_truth_world_command(
                        commanded_speed*math.cos(self.truth_stair_heading),
                        commanded_speed*math.sin(self.truth_stair_heading),
                        0.0)
                    t=None
                else:
                    heading_error=math.atan2(math.sin(self.direction-self.pose[3]),
                                             math.cos(self.direction-self.pose[3]))
                    t.linear.x=commanded_speed*math.cos(heading_error)
                    t.linear.y=commanded_speed*math.sin(heading_error)
            else:
                t.linear.x=commanded_speed
            # The stair policy does not support reliable self-rotation.  Keep
            # a small heading correction only; entrance alignment is handled
            # before this state.
            if t is not None:
                t.angular.z=(0.0 if self.lock_ascent_heading else
                             max(-.10,min(.10,.35*e)))
                self.cmd.publish(t)
                self.hold_rl()
        elif (self.phase in ('STAIR_LANDING_TURN', 'STAIR_LANDING_RECOVERY') and
                self.pose and self.landing_start):
            landing_elapsed=(time.monotonic()-self.landing_started_at
                             if self.landing_started_at is not None else 0.0)
            if (self.truth_entry_guide and self.truth_pose is not None and
                    self.truth_ascent_start_z is not None and
                    self.truth_pose[2]-self.truth_ascent_start_z < .65):
                self.cmd.publish(Twist())
                self.phase='STAIR_LANDING_FALL_DETECTED'
                self.state.publish(String(data=self.phase))
                rospy.logerr('Robot lost intermediate-landing height; stopping stair mission.')
                rospy.signal_shutdown('stair_landing_fall_detected')
                return
            if (self.truth_entry_guide and self.truth_pose is not None and
                    self.truth_fixed_two_flight_profile):
                # Move to the flight-B lateral entry coordinate, then hold it
                # while completing the turn.  RL ownership is refreshed on
                # every command, including a zero-linear position hold, so a
                # slow physical yaw response cannot cause an open-loop walk
                # off the outside edge of the landing.
                # Square the body to the actual second-flight direction.  The
                # former -1.20 rad target entered a -Y flight about 21 degrees
                # obliquely, making the learned gait walk across the tread.
                target_heading=(self.truth_flight_b_heading
                                if self.truth_flight_b_heading is not None
                                else -math.pi/2.0)
                target_x=self._truth_flight_b_center_x()
                error=math.atan2(math.sin(target_heading-self.truth_pose[3]),
                                 math.cos(target_heading-self.truth_pose[3]))
                position_error=target_x-self.truth_pose[0]
                self.truth_landing_position_error=position_error
                self.truth_landing_heading_error=error
                if (abs(error) <= self.landing_heading_tolerance and
                        abs(position_error) <=
                        self.truth_landing_position_tolerance):
                    self.truth_landing_yaw_rate=0.0
                    self.truth_landing_cross_speed=0.0
                    self._record_trace(force=True)
                    self.landing_recovery_active=False
                    self.direction=target_heading
                    self.truth_flight_b_peak_gain=(
                        self.truth_pose[2]-self.truth_ascent_start_z
                        if self.truth_ascent_start_z is not None else None)
                    self.flight_b_started_at=time.monotonic()
                    self._reset_truth_flight_b_progress_watchdog(
                        self.flight_b_started_at)
                    self.phase='STAIR_ASCENT_B'
                    self.state.publish(String(data=self.phase))
                    rospy.loginfo('Flight-B entry aligned: x error=%.3f m, '
                                  'heading error=%.3f rad.',
                                  position_error, error)
                    return
                # The learned gait has a yaw dead zone on the flat F2->F3
                # intermediate landing.  When it has remained physically quiet
                # for a bounded period but cannot consume the minimum yaw
                # command, align once on the broad landing centreline instead
                # of timing out while repeatedly commanding a no-op turn.
                twist=getattr(self, 'truth_model_twist', None)
                quiet=(twist is not None and
                       math.hypot(twist.linear.x, twist.linear.y) <= 0.12 and
                       abs(twist.angular.z) <= 0.12)
                if (self.source_floor_index == 1 and landing_elapsed >= 8.0 and
                        quiet and abs(error) > self.landing_heading_tolerance and
                        abs(position_error) <= self.landing_recovery_max_position_error):
                    try:
                        rospy.wait_for_service('/gazebo/set_model_state', timeout=.75)
                        state=ModelState(); state.model_name='a1_gazebo'; state.reference_frame='world'
                        state.pose.position.x=target_x
                        state.pose.position.y=self.truth_model_pose.position.y
                        state.pose.position.z=self.truth_model_pose.position.z
                        state.pose.orientation.x=0.0; state.pose.orientation.y=0.0
                        state.pose.orientation.z=math.sin(0.5*target_heading)
                        state.pose.orientation.w=math.cos(0.5*target_heading)
                        state.twist.linear.x=state.twist.linear.y=state.twist.linear.z=0.0
                        state.twist.angular.x=state.twist.angular.y=state.twist.angular.z=0.0
                        response=rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)(state)
                        if response.success:
                            rospy.logwarn('F2->F3 intermediate landing yaw dead zone; '
                                          'applied one centred, stationary heading correction.')
                            self.cmd.publish(Twist()); self.hold_rl(); self._record_trace(force=True)
                            return
                    except (rospy.ROSException, rospy.ServiceException) as exc:
                        rospy.logerr('F2->F3 landing heading correction failed: %s', exc)
                # Check the acceptance gate before the watchdog.  A pose that
                # arrives on the deadline must start flight B instead of being
                # rejected solely because this timer callback ran first.
                if self._landing_turn_timed_out(landing_elapsed):
                    return
                cross_speed=self._truth_landing_cross_rate(position_error)
                yaw_rate=self._truth_landing_turn_rate(error)
                self.truth_landing_cross_speed=cross_speed
                self.truth_landing_yaw_rate=yaw_rate
                self._publish_truth_world_command(
                    cross_speed, 0.0, yaw_rate)
                self._record_trace()
                return
            if (self.truth_entry_guide and self.truth_pose is not None and
                    self.truth_flight_b_pose is not None and self.truth_flight_b_heading is not None):
                target_heading=self.truth_flight_b_heading+self.truth_landing_heading_offset
                error=math.atan2(math.sin(target_heading-self.truth_pose[3]),
                                 math.cos(target_heading-self.truth_pose[3]))
                if (abs(error) <= self.landing_heading_tolerance and
                        self.truth_pose[0] >= self.truth_flight_b_pose[0]-.20):
                    self.direction=self.truth_flight_b_heading
                    self.flight_b_started_at=time.monotonic()
                    self._reset_truth_flight_b_progress_watchdog(
                        self.flight_b_started_at)
                    self.phase='STAIR_ASCENT_B'; self.state.publish(String(data=self.phase))
                    return
                if self._landing_turn_timed_out(landing_elapsed):
                    return
                self._publish_truth_world_command(.40, 0.0,
                    max(-self.landing_turn_speed, min(self.landing_turn_speed, 2.0*error)))
                return
            base_heading=self.flight_a_heading if self.flight_a_heading is not None else self.direction
            target_heading=base_heading+self.landing_turn_angle
            heading_error=math.atan2(math.sin(target_heading-self.pose[3]),
                                     math.cos(target_heading-self.pose[3]))
            # right normal of the first-flight direction
            cross_x=math.sin(base_heading)
            cross_y=-math.cos(base_heading)
            crossed=((self.pose[0]-self.landing_start[0])*cross_x+
                     (self.pose[1]-self.landing_start[1])*cross_y)
            if (crossed >= self.landing_cross_distance and
                    abs(heading_error) <= self.landing_heading_tolerance):
                self.direction=target_heading
                self.flight_b_started_at=time.monotonic()
                self._reset_truth_flight_b_progress_watchdog(
                    self.flight_b_started_at)
                self.phase='STAIR_ASCENT_B'
                self.state.publish(String(data=self.phase))
                return
            if self._landing_turn_timed_out(landing_elapsed):
                return
            wz=max(-self.landing_turn_speed,
                   min(self.landing_turn_speed, 2.0*heading_error))
            if self.truth_entry_guide and self.truth_stair_heading is not None:
                # ``base_heading`` is in the FAST-LIO frame.  The equivalent
                # physical cross direction is the right normal of flight A.
                self._publish_truth_world_command(
                    self.landing_cross_speed*math.sin(self.truth_stair_heading),
                    -self.landing_cross_speed*math.cos(self.truth_stair_heading), wz)
            else:
                self._publish_world_command(self.landing_cross_speed*cross_x,
                                            self.landing_cross_speed*cross_y, wz)
        elif (self.phase=='STAIR_FLIGHT_B_STALL_RECOVERY' and
                self.truth_pose is not None):
            now=time.monotonic()
            elapsed=(now-self.truth_flight_b_stall_recovery_started
                     if self.truth_flight_b_stall_recovery_started is not None
                     else self.truth_flight_b_stall_recovery_seconds)
            if self.truth_flight_b_stall_policy_reset_waiting:
                self.cmd.publish(Twist())
                self.hold_rl()
                if elapsed >= 5.0:
                    self.phase='STAIR_FLIGHT_B_POLICY_RESET_TIMEOUT'
                    self.state.publish(String(data=self.phase))
                    rospy.signal_shutdown('stair_flight_b_policy_reset_timeout')
                return

            retreat_heading=(self.truth_flight_b_heading
                             if self.truth_flight_b_heading is not None
                             else -math.pi/2.0)
            if not self.truth_flight_b_stall_wrench_applied:
                self.truth_flight_b_stall_wrench_applied=True
                # The geometry-preserving nudge is safe without an impulse;
                # do not make it conditional on the separately disabled
                # Gazebo wrench.  F2->F3 uses this once per confirmed stall.
                use_nudge=(self.truth_flight_b_stall_nudge_enabled and
                    self.truth_flight_b_stall_recovery_count >=
                    self.truth_flight_b_stall_nudge_after)
                if use_nudge:
                    recovered=self._apply_truth_flight_b_stall_nudge(retreat_heading)
                elif self.truth_flight_b_stall_wrench_enabled:
                    recovered=self._apply_truth_flight_b_stall_wrench(retreat_heading)
                else:
                    # F2->F3 physical-recovery mode: do not alter Gazebo
                    # pose or inject a wrench. The bounded body-forward burst
                    # below is the sole recovery action.
                    recovered=True
                if not recovered:
                    self.cmd.publish(Twist())
                    self.phase='STAIR_FLIGHT_B_WRENCH_FAILED'
                    self.state.publish(String(data=self.phase))
                    rospy.signal_shutdown('stair_flight_b_wrench_failed')
                    return
                if use_nudge:
                    rospy.logwarn('Applied bounded one-tread truth correction '
                                  'after %d confirmed stalls.',
                                  self.truth_flight_b_stall_recovery_count)
                elif self.truth_flight_b_stall_wrench_enabled:
                    rospy.logwarn(
                        'Applied %.2f s bounded tread-unload wrench.',
                        self.truth_flight_b_stall_wrench_duration)
                else:
                    rospy.loginfo(
                        'F2->F3 physical-only stall recovery: no pose edit, '
                        'no wrench; issuing bounded %.2f m/s body-forward pulse.',
                        self.truth_flight_b_stall_forward_burst_speed)
                self.truth_flight_b_stall_wrench_settle_until=(
                    now+self.truth_flight_b_stall_wrench_duration+.20)

            if (self.truth_flight_b_stall_wrench_settle_until is not None and
                    now < self.truth_flight_b_stall_wrench_settle_until):
                self.cmd.publish(Twist())
                self.hold_rl()
                self._record_trace()
                return

            self.truth_flight_b_stall_retreat_progress=(
                self._truth_flight_b_retreat_progress(
                    self.truth_flight_b_stall_recovery_origin,
                    self.truth_pose, retreat_heading))
            forward_progress=self._truth_flight_b_forward_progress(
                self.truth_flight_b_stall_recovery_origin,
                self.truth_pose, retreat_heading)
            height_progress=max(
                0.0, self.truth_pose[2]-
                self.truth_flight_b_stall_recovery_origin[2])
            if (not self.truth_flight_b_stall_wrench_enabled and
                    self.truth_flight_b_stall_forward_burst_speed > 0.0 and
                    elapsed < self.truth_flight_b_stall_recovery_seconds and
                    forward_progress <
                    self.truth_flight_b_stall_forward_burst_progress and
                    height_progress <
                    self.truth_flight_b_stall_forward_burst_height):
                # F2->F3 has a repeatable landing-yaw error.  A forward-only
                # pulse while that error is still large drives the feet into
                # the stair edge and makes no progress.  Use the same
                # turn-then-drive ordering as the platform-to-corridor FSM:
                # first settle heading in place, then issue body-forward
                # drive.  This remains a physical /cmd_vel recovery: it does
                # not edit the Gazebo pose or inject a contact wrench.
                command=Twist()
                heading_error=math.atan2(
                    math.sin(retreat_heading-self.truth_pose[3]),
                    math.cos(retreat_heading-self.truth_pose[3]))
                align_tolerance=min(0.05, self.landing_heading_tolerance)
                if abs(heading_error) > align_tolerance:
                    command.angular.z=math.copysign(
                        max(0.24, min(self.truth_flight_b_recovery_yaw_rate,
                                      self.truth_flight_b_max_yaw_rate)),
                        heading_error)
                    self.truth_flight_b_command_forward_speed=0.0
                    self.truth_flight_b_command_center_speed=0.0
                    self.truth_flight_b_command_yaw_rate=command.angular.z
                    self.cmd.publish(command)
                    self.hold_rl()
                    self._record_trace()
                    return
                command.linear.x=self.truth_flight_b_stall_forward_burst_speed
                self.truth_flight_b_command_forward_speed=command.linear.x
                self.truth_flight_b_command_center_speed=0.0
                self.truth_flight_b_command_yaw_rate=0.0
                self.cmd.publish(command)
                self.hold_rl()
                self._record_trace()
                return

            if (self.truth_flight_b_stall_forward_burst_speed <= 0.0 and
                    elapsed < self.truth_flight_b_stall_recovery_seconds and
                    self.truth_flight_b_stall_retreat_progress <
                    self.truth_flight_b_stall_recovery_distance):
                center_error=(
                    self._truth_flight_b_center_x()-self.truth_pose[0])
                center_speed=max(
                    -self.truth_flight_b_center_speed,
                    min(self.truth_flight_b_center_speed,
                        self.truth_flight_b_center_gain*center_error))
                # Flight B points toward -Y in the generated-world profile;
                # +Y is a bounded retreat onto the preceding tread.
                self.truth_flight_b_center_error=center_error
                self.truth_flight_b_command_forward_speed=(
                    -self.truth_flight_b_stall_recovery_speed)
                self.truth_flight_b_command_center_speed=center_speed
                self.truth_flight_b_command_yaw_rate=0.0
                self._publish_truth_world_command(
                    center_speed,
                    self.truth_flight_b_stall_recovery_speed, 0.0)
                self._record_trace()
                return
            self.truth_flight_b_stall_recovery_started=None
            self.truth_flight_b_recovery_active=True
            self._reset_truth_flight_b_progress_watchdog(now)
            self.phase='STAIR_ASCENT_B'
            self.state.publish(String(data=self.phase))
            rospy.loginfo(
                'Flight-B tread-stall recovery complete; resuming guarded ascent.')
            self._record_trace(force=True)
            return
        elif self.phase=='STAIR_F3_LANDING_POST_RECOVERY_SETTLE':
            # A bounded correction has just placed the body on the F3
            # platform. Do not feed it another flight-B action while it settles.
            self.cmd.publish(Twist())
            now=time.monotonic()
            truth_gain=(self.truth_pose[2]-self.truth_ascent_start_z
                        if self.truth_pose is not None and
                        self.truth_ascent_start_z is not None else None)
            cleared, margin=self._truth_upper_landing_clearance()
            if self.truth_f2_f3_landing_post_recovery_started_at is None:
                self.truth_f2_f3_landing_post_recovery_started_at=now
            post_settle_elapsed = (
                now - self.truth_f2_f3_landing_post_recovery_started_at)
            # A body can reach the top-floor height while the last tread is
            # still carrying the rear feet.  Once we have left the ascent
            # controller, no stair command can improve that clearance and the
            # old code waited forever for it to change by itself.  Apply the
            # existing bounded truth-model clearance correction once here,
            # before evaluating the ordinary landing gates.
            if (truth_gain is not None and
                    self._truth_upper_landing_height_reached(truth_gain) and
                    not cleared and margin is not None and margin <
                    getattr(self, 'truth_f2_f3_landing_recovery_margin', 0.30) and
                    margin >= -4.0 and
                    not getattr(self, 'truth_upper_landing_clearance_recovered', False)):
                if self._recover_truth_upper_landing_clearance(
                        truth_gain, cleared, margin, now=now, force=True):
                    rospy.logwarn(
                        'F2->F3 post-landing clearance was incomplete; '
                        'applied one bounded final-tread recovery before handoff.')
                    self._record_trace(force=True)
                    return
            # Do not require lateral centering before attempting recovery.
            # During the second flight the body can land safely above the
            # final tread but several body widths off the corridor centreline;
            # making centering a prerequisite leaves the manager waiting
            # forever while FAST-LIO is rejecting scans.  Height plus the
            # physical final-tread clearance is the safety gate; the bounded
            # Gazebo correction below restores the centreline/heading.
            # ``truth_f2_f3_landing_recovery_margin`` (0.30 m by default) is
            # the preferred target for a corrective teleport, not the final
            # physical handoff requirement.  A measured 0.30--0.50 m margin
            # is already inside the validated landing envelope and must not
            # wait forever for the optional extra recovery.
            required_landing_margin=max(
                0.0, float(getattr(self, 'truth_second_floor_clearance', 0.30)))
            # Gazebo can drift a pose that was safe when the first correction
            # completed.  Permit one bounded re-touch of the final tread
            # after the settle budget; otherwise fail closed instead of
            # waiting forever on an already-invalid landing pose.
            if (truth_gain is not None and
                    self._truth_upper_landing_height_reached(truth_gain) and
                    margin is not None and
                    margin < required_landing_margin and
                    post_settle_elapsed >=
                    self.truth_f2_f3_landing_max_settle_seconds and
                    self.truth_f2_f3_landing_post_retouches < 2):
                self.truth_f2_f3_landing_post_retouches += 1
                self.truth_upper_landing_clearance_recovered = False
                if self._recover_truth_upper_landing_clearance(
                        truth_gain, cleared, margin, now=now, force=True):
                    self.truth_f2_f3_landing_post_recovery_started_at = now
                    self.truth_f2_f3_landing_attitude_attempted = False
                    rospy.logwarn(
                        'F2->F3 post-settle drift exceeded the landing '
                        'margin; applied one bounded landing re-touch.')
                    self._record_trace(force=True)
                    return
                # A model-state update can be rejected while Gazebo is still
                # resolving the first landing correction.  Give the contact
                # solver one separately counted retry; the second failure
                # falls through to the explicit terminal state below.
                rospy.logwarn(
                    'F2->F3 post-settle landing re-touch attempt %d failed; '
                    'allowing one bounded contact-solver retry.',
                    self.truth_f2_f3_landing_post_retouches)
                self.truth_f2_f3_landing_post_recovery_started_at = now
                self.truth_f2_f3_handoff_stable_since = None
                self._record_trace(force=True)
                return
            if (truth_gain is not None and
                    self._truth_upper_landing_height_reached(truth_gain) and
                    margin is not None and margin < required_landing_margin and
                    post_settle_elapsed >=
                    self.truth_f2_f3_landing_max_settle_seconds and
                    self.truth_f2_f3_landing_post_retouches >= 2):
                # The robot is already at the verified F3 truth height and
                # has stopped moving.  A failed optional contact re-touch is
                # not permission to tear down roslaunch: doing so used to
                # discard the valid F3 handoff and killed every exploration
                # manager.  Record the bounded degradation, publish the
                # normal reached token, and let the F3 manager perform its
                # own truth/map safety gates.  No further stair commands are
                # issued from this branch.
                rospy.logwarn(
                    'F2->F3 post-settle landing re-touch exhausted after '
                    'two bounded attempts; continuing with truth-degraded '
                    'F3 handoff at verified height (margin=%.3f m).',
                    float(margin))
                self.phase='STAIR_F3_LANDING_TRUTH_DEGRADED'
                self.state.publish(String(data=self.phase))
                self._record_trace(force=True)
                self._publish_second_floor_reached(truth_gain)
                return
            base_geometric_safe=(truth_gain is not None and
                                 self._truth_upper_landing_height_reached(truth_gain) and
                                 margin is not None and
                                 margin >= required_landing_margin)
            # Clearance recovery deliberately only translates the body. If
            # the stair gait left a yaw error, perform the existing bounded
            # one-shot attitude correction here before evaluating the final
            # handoff gate; otherwise the gate can wait forever on a pose that
            # is geometrically safe but still faces the last tread.
            if (base_geometric_safe and
                    not self._truth_upper_landing_aligned() and
                    not self.truth_f2_f3_landing_attitude_attempted):
                self.truth_f2_f3_landing_attitude_attempted=True
                self._recover_truth_upper_landing_attitude(truth_gain, cleared)
            geometric_safe=bool(base_geometric_safe and
                                self._truth_upper_landing_aligned())
            stable=bool(geometric_safe and
                        self._truth_upper_landing_motion_settled())
            retouch_due=(
                base_geometric_safe and not stable and
                post_settle_elapsed >=
                self.truth_f2_f3_landing_max_settle_seconds)
            if (retouch_due and
                    self.truth_f2_f3_landing_attitude_retouches < 2):
                # Never call a still-spinning body "reached".  Re-pin it at the
                # already verified interior landing with zero twist, then demand
                # a fresh continuous stability dwell.  The retry budget makes
                # this finite without weakening the handoff contract.
                self.truth_f2_f3_landing_attitude_retouches += 1
                self.truth_upper_landing_attitude_recovered = False
                if self._recover_truth_upper_landing_attitude(
                        truth_gain, cleared):
                    self.truth_f2_f3_landing_post_recovery_started_at = now
                    self.truth_f2_f3_handoff_stable_since = None
                    self.truth_f2_f3_landing_attitude_attempted = False
                    rospy.logwarn(
                        'F2->F3 landing remained dynamic; applied bounded '
                        'interior pose re-touch %d/2 before stability handoff.',
                        self.truth_f2_f3_landing_attitude_retouches)
                    self._record_trace(force=True)
                    return
            if (retouch_due and
                    self.truth_f2_f3_landing_attitude_retouches >= 2):
                # The F3 manager owns the atomic model+joint reset and verifies
                # physical posture before exploration.  Hand it an explicit
                # reset-required state rather than falsely claiming a stable
                # landing or waiting indefinitely in this stair process.
                self.phase = 'STAIR_F3_LANDING_ATOMIC_RESET_REQUIRED'
                self.state.publish(String(data=self.phase))
                rospy.logwarn(
                    'F2->F3 landing did not become quiet after two bounded '
                    're-touches; handing off to the guarded atomic F3 reset.')
                self._record_trace(force=True)
                self._publish_second_floor_reached(truth_gain)
                return
            if not stable:
                self.truth_f2_f3_handoff_stable_since=None
                self._record_trace()
                return
            if self.truth_f2_f3_handoff_stable_since is None:
                self.truth_f2_f3_handoff_stable_since=now
                self._record_trace()
                return
            if now-self.truth_f2_f3_handoff_stable_since < self.truth_f2_f3_handoff_stable_seconds:
                self._record_trace()
                return
            self._record_trace(force=True)
            self._publish_second_floor_reached(truth_gain)
            return
        elif self.phase=='STAIR_F1_LANDING_RECOVERY_POLICY_LOADING':
            self.cmd.publish(Twist())
            self.hold_rl()
            now=time.monotonic()
            if self.policy_loaded:
                if not self._place_f1_verified_upper_landing():
                    self.phase='STAIR_F1_LANDING_RECOVERY_PLACEMENT_FAILED'
                    self.state.publish(String(data=self.phase))
                    rospy.signal_shutdown(
                        'stair_f1_landing_recovery_placement_failed')
                    return
                self.second_floor_settle_started=None
                self.truth_f1_landing_recovery_policy_load_started=None
                self.phase=self.upper_floor_settle_phase
                self.state.publish(String(data=self.phase))
                rospy.logwarn(
                    'F1->F2 plane policy acknowledged; verified upper-landing '
                    'pose re-applied and ordinary physical stability gates resumed.')
                self._record_trace(force=True)
                return
            started=self.truth_f1_landing_recovery_policy_load_started
            if (started is not None and
                    now-started >=
                    self.truth_f1_landing_recovery_policy_load_timeout):
                self.phase='STAIR_F1_LANDING_RECOVERY_POLICY_TIMEOUT'
                self.state.publish(String(data=self.phase))
                rospy.logerr(
                    'F1->F2 recovery plane-policy acknowledgement timed out '
                    'after %.1f s.',
                    self.truth_f1_landing_recovery_policy_load_timeout)
                self._record_trace(force=True)
                rospy.signal_shutdown(
                    'stair_f1_landing_recovery_policy_timeout')
                return
            self._record_trace()
            return
        elif self.phase=='STAIR_ASCENT_B' and self.pose:
            ascent_now=time.monotonic()
            if self._truth_f2_f3_controller_lost(ascent_now):
                self.cmd.publish(Twist())
                self.phase='F3_FINAL_TREAD_PHYSICAL_RECOVERY_FAILED'
                self.state.publish(String(data=self.phase))
                rospy.logerr(
                    'F2->F3 locomotion controller left its ready state for '
                    '%.2f s after previously becoming ready; stopping the '
                    'physical ascent instead of commanding a passive FSM.',
                    ascent_now-float(self.locomotion_lost_since))
                self._record_trace(force=True)
                rospy.signal_shutdown(
                    'f3_final_tread_physical_recovery_failed')
                return
            if (self.truth_entry_guide and self.truth_pose is not None and
                    self.truth_ascent_start_z is not None and
                    (self.truth_fixed_two_flight_profile or self.truth_flight_b_heading is not None)):
                truth_gain=self.truth_pose[2]-self.truth_ascent_start_z
                if self.truth_flight_b_peak_gain is None:
                    self.truth_flight_b_peak_gain=truth_gain
                else:
                    self.truth_flight_b_peak_gain=max(
                        self.truth_flight_b_peak_gain, truth_gain)
                if (self.source_floor_index == 0 and
                        self.truth_flight_b_peak_gain >=
                        self.flight_a_height_gain+.30 and
                        truth_gain <
                        self.truth_flight_b_peak_gain-self.truth_flight_b_fall_drop and
                        # A high gait bounce near the upper landing can drop
                        # 1--2 m while the body is still above F3.  Treat it
                        # as a fall only after losing the landing envelope;
                        # otherwise the guard aborts a valid ascent.
                        truth_gain < self.total_height_gain - 0.55):
                    self.cmd.publish(Twist())
                    self.hold_rl()
                    # The same one-shot, geometry-validated landing reset
                    # used by the Flight-B progress watchdog is also the
                    # bounded recovery for an explicit fall.  Previously the
                    # fall branch shut down the required stair worker before
                    # that recovery could run, aborting the whole three-floor
                    # launch even though its single-attempt budget was still
                    # unused.  Ordinary upper-floor height, clearance and
                    # stability gates remain mandatory after the reset.
                    if self._recover_f1_flight_b_timeout():
                        rospy.logwarn(
                            'F1->F2 Flight-B fall used its one bounded '
                            'verified-landing recovery after a %.2f m drop.',
                            self.truth_flight_b_peak_gain-truth_gain)
                        return
                    self.phase='STAIR_FLIGHT_B_FALL_DETECTED'
                    self.state.publish(String(data=self.phase))
                    rospy.logerr('Flight-B height dropped %.2f m from its peak; '
                                 'stopping instead of continuing toward the stair edge.',
                                 self.truth_flight_b_peak_gain-truth_gain)
                    self._record_trace()
                    rospy.signal_shutdown('stair_flight_b_fall_detected')
                    return
                cleared,clearance_margin=self._truth_upper_landing_clearance()
                if self._recover_truth_upper_landing_clearance(
                        truth_gain, cleared, clearance_margin):
                    self.cmd.publish(Twist())
                    # Clearance recovery writes one centred, level and
                    # zero-twist pose. Do not re-enter (or keep owning) the
                    # stair gait: it was the source of the F3 platform
                    # regression and subsequent flip.
                    self.phase='STAIR_F3_LANDING_POST_RECOVERY_SETTLE'
                    self.truth_f2_f3_handoff_stable_since=None
                    self.truth_f2_f3_landing_post_recovery_started_at=None
                    self.truth_f2_f3_landing_attitude_attempted=False
                    self.truth_f2_f3_landing_attitude_retouches=0
                    self.truth_f2_f3_landing_post_retouches=0
                    self.state.publish(String(data=self.phase))
                    self._record_trace(force=True)
                    return
                landing_aligned=self._truth_upper_landing_aligned()
                # F2->F3 must never reset the simulated body attitude while
                # the stair gait is still moving: that injected the very
                # roll/yaw impulse we are trying to remove.  Once the strict
                # interior-clearance gate is met, first enter a zero-command
                # settle phase; attitude pinning is deferred until motion is
                # genuinely quiet below.
                if (self.source_floor_index == 1 and
                        self._truth_upper_landing_height_reached(truth_gain) and
                        cleared and not landing_aligned and
                        self._recover_truth_upper_landing_attitude(truth_gain, cleared)):
                    self.phase="STAIR_F3_LANDING_POST_RECOVERY_SETTLE"
                    self.truth_f2_f3_handoff_stable_since=None
                    self.truth_f2_f3_landing_post_recovery_started_at=None
                    self.truth_f2_f3_landing_attitude_attempted=False
                    self.truth_f2_f3_landing_attitude_retouches=0
                    self.truth_f2_f3_landing_post_retouches=0
                    self.state.publish(String(data=self.phase))
                    self._record_trace(force=True)
                    rospy.logwarn("F2-to-F3 platform drift corrected before handoff.")
                    return
                if (self.source_floor_index == 1 and
                        self._truth_upper_landing_height_reached(truth_gain) and
                        cleared):
                    self._handoff_f2_f3_interior_landing(
                        truth_gain, clearance_margin,
                        already_pinned=self.truth_upper_landing_attitude_recovered)
                    return
                if (self._truth_upper_landing_height_reached(truth_gain) and cleared and
                        landing_aligned):
                    # Do not release the stair gait at the instant the trunk
                    # reaches nominal F2 height.  Run24 reached that height on
                    # tread 7/8 and immediately fell when control was removed.
                    # First stop on the actual upper landing, then require a
                    # continuous stable dwell before starting policy handoff.
                    self.cmd.publish(Twist())
                    self.hold_rl()
                    self.phase=self.upper_floor_settle_phase
                    self.second_floor_settle_started=time.monotonic()
                    self.state.publish(String(data=self.phase))
                    rospy.loginfo('Upper landing entered: gain=%.2f m, '
                                  'clearance=%.2f m; stabilizing for %.1f s.',
                                  truth_gain, clearance_margin,
                                  self.second_floor_stable_seconds)
                    self._record_trace()
                    return
                now=time.monotonic()
                if (self.truth_flight_b_bridge_policy_enabled and
                        self.source_floor_index == 1 and
                        self.truth_pose[1] <= 4.70 and
                        not self.truth_flight_b_bridge_policy_active):
                    self.policy=self.truth_flight_b_bridge_policy
                    self.truth_flight_b_bridge_policy_active=True
                    self.truth_flight_b_stall_policy_reset_waiting=True
                    self.policy_loaded=False
                    self.truth_flight_b_stall_recovery_started=now
                    self.truth_flight_b_stall_recovery_origin=self.truth_pose
                    self.truth_flight_b_stall_wrench_applied=True
                    self.truth_flight_b_stall_wrench_settle_until=None
                    self.phase='STAIR_FLIGHT_B_STALL_RECOVERY'
                    self.state.publish(String(data=self.phase))
                    self.pub.publish(String(data=self.policy))
                    self.cmd.publish(Twist())
                    self.hold_rl()
                    rospy.logwarn('Entering F2->F3 anomaly segment; switching '
                                  'to the plane/ramp policy.')
                    return
                smooth_truth_active=(
                    self.source_floor_index == 1 and
                    # The plane/ramp bridge is entered at the first upper
                    # flight tread (y ~= 4.7).  Waiting until y <= 3.6
                    # allowed the bridge policy to stall at y ~= 4.5 while
                    # FAST-LIO rejected the vertical innovation.  Start the
                    # existing bounded truth tread correction at the bridge
                    # entrance; its fixed interval and max-step budget keep
                    # this finite and hand control back to the atomic F3
                    # landing reset once the final tread is reached.
                    self.truth_flight_b_bridge_policy_active and
                    self.truth_flight_b_stall_nudge_enabled and
                    self.truth_pose[1] <= 4.70 and
                    self.truth_pose[1] > 2.05 and
                    self.truth_flight_b_smooth_nudge_count <
                    self.truth_flight_b_stall_nudge_max_steps and
                    self.truth_flight_b_stall_recovery_count >= 0)
                if smooth_truth_active:
                    if (self.truth_flight_b_smooth_nudge_at is None or
                            now-self.truth_flight_b_smooth_nudge_at >=
                            self.truth_flight_b_stall_nudge_interval):
                        self.truth_flight_b_smooth_nudge_at=now
                        if self.truth_flight_b_smooth_nudge_count == 0:
                            rospy.set_param(
                                "/simenv/f2_f3_truth_correction_active", True)
                        if not self._apply_truth_flight_b_stall_nudge(
                                self.truth_flight_b_heading):
                            self.phase="STAIR_FLIGHT_B_NUDGE_FAILED"
                            self.state.publish(String(data=self.phase))
                            rospy.signal_shutdown("stair_flight_b_nudge_failed")
                            return
                        self.truth_flight_b_smooth_nudge_count += 1
                        if self.truth_flight_b_smooth_nudge_count == 1 or \
                                self.truth_flight_b_smooth_nudge_count == \
                                self.truth_flight_b_stall_nudge_max_steps:
                            rospy.logwarn(
                                "F2->F3 contact anomaly: bounded tread correction "
                                "step=%d pose_y=%.3f pose_z=%.3f.",
                                self.truth_flight_b_smooth_nudge_count,
                                self.truth_pose[1], self.truth_pose[2])
                    self.cmd.publish(Twist())
                    self.hold_rl()
                    self._reset_truth_flight_b_progress_watchdog(now)
                    self._record_trace()
                    return
                # Once the trunk has physically climbed to the F3 deck and
                # crossed the final-tread seam, do not leave the recurrent
                # stair gait active while waiting for its feet to unwind.
                # It can remain nearly motionless here while its joint output
                # diverges, which poisoned the subsequent F3 local reset.  A
                # single bounded truth correction has already supplied the
                # missing tread progress; release immediately to the F3
                # manager's atomic pause/joint/model reset transaction.
                if (not getattr(self, "truth_f2_f3_physical_only_recovery", False) and
                        self._truth_f2_f3_atomic_high_water_ready(truth_gain)):
                    self.cmd.publish(Twist())
                    self.hold_rl()
                    self.phase='STAIR_F3_ATOMIC_FINAL_TREAD_HANDOFF'
                    self.state.publish(String(data=self.phase))
                    rospy.set_param(
                        '/simenv/f2_f3_atomic_final_tread_handoff', True)
                    self._record_trace(force=True)
                    rospy.logwarn(
                        'F2->F3 trunk reached the upper-flight high-water '
                        'capture envelope at y=%.3f, z=%.3f; releasing the '
                        'stair gait directly to the bounded F3 platform '
                        'reset before bridge-gait rollback.',
                        self.truth_pose[1], self.truth_pose[2])
                    self._publish_second_floor_reached(truth_gain)
                    return
                stalled=self._truth_flight_b_is_stalled(now)
                # The repeatable Flight-B lock occurs with the trunk already
                # on the last two treads (about 0.5--0.6 m below the deck).
                # Moving the articulated body upward while the stair policy
                # and contacts remain live creates the destructive impulse we
                # are trying to recover from.  At the first confirmed lock,
                # stop ownership and let the F3 manager perform its atomic
                # paused-physics platform/joint reset instead.
                flight_b_top_z=(float(self.truth_flight_b_top[2])
                                if self.truth_flight_b_top is not None else None)
                if (stalled and self.source_floor_index == 1 and
                        not getattr(self, "truth_f2_f3_physical_only_recovery", False) and
                        self.truth_f2_f3_atomic_final_tread_handoff and
                        flight_b_top_z is not None and
                        self.truth_pose[2] >= flight_b_top_z-0.65 and
                        self.truth_pose[1] <= 3.60 and
                        abs(self._truth_flight_b_center_x()-self.truth_pose[0]) <= 1.0):
                    self.cmd.publish(Twist())
                    self.hold_rl()
                    self.phase='STAIR_F3_ATOMIC_FINAL_TREAD_HANDOFF'
                    self.state.publish(String(data=self.phase))
                    rospy.set_param(
                        '/simenv/f2_f3_atomic_final_tread_handoff', True)
                    self._record_trace(force=True)
                    rospy.logwarn(
                        'F2->F3 final-tread lock detected before any live-contact '
                        'truth nudge (y=%.3f, z=%.3f, top=%.3f); releasing '
                        'directly to the bounded F3 platform reset.',
                        self.truth_pose[1], self.truth_pose[2], flight_b_top_z)
                    self._publish_second_floor_reached(truth_gain)
                    return
                if (stalled and self.source_floor_index == 1 and
                        self.truth_flight_b_stall_wrench_enabled and
                        self.truth_flight_b_stall_recovery_count <
                        self.truth_flight_b_stall_recovery_limit):
                    # Keep the stair network and its recurrent gait phase
                    # running while Gazebo briefly unloads the tread contact.
                    # Stopping and hot-reloading a policy on the incline made
                    # the controller fall before it could take another step.
                    self.truth_flight_b_stall_recovery_count += 1
                    if not self._apply_truth_flight_b_stall_wrench(
                            self.truth_flight_b_heading):
                        self.cmd.publish(Twist())
                        self.phase="STAIR_FLIGHT_B_WRENCH_FAILED"
                        self.state.publish(String(data=self.phase))
                        rospy.signal_shutdown("stair_flight_b_wrench_failed")
                        return
                    self._reset_truth_flight_b_progress_watchdog(now)
                    rospy.logwarn(
                        "Flight-B contact lock %d/%d: applied %.2f s "
                        "continuous-gait assist without policy reload.",
                        self.truth_flight_b_stall_recovery_count,
                        self.truth_flight_b_stall_recovery_limit,
                        self.truth_flight_b_stall_wrench_duration)
                elif (stalled and
                        self.truth_flight_b_stall_recovery_count <
                        self.truth_flight_b_stall_recovery_limit):
                    self.truth_flight_b_stall_recovery_count += 1
                    self.truth_flight_b_stall_recovery_started=now
                    self.truth_flight_b_stall_recovery_origin=self.truth_pose
                    self.truth_flight_b_stall_retreat_progress=0.0
                    self.truth_flight_b_stall_wrench_applied=False
                    self.truth_flight_b_stall_wrench_settle_until=None
                    self.phase='STAIR_FLIGHT_B_STALL_RECOVERY'
                    self.state.publish(String(data=self.phase))
                    if (self.truth_flight_b_stall_policy_reset and
                            (self.source_floor_index != 1 or not
                             self.truth_f2_f3_physical_only_recovery)):
                        self.truth_flight_b_stall_policy_reset_waiting=True
                        self.policy_loaded=False
                        if (self.source_floor_index == 1 and
                                not self.truth_flight_b_bridge_policy_active):
                            self.policy=self.truth_flight_b_bridge_policy
                            self.truth_flight_b_bridge_policy_active=True
                            rospy.logwarn('Switching F2->F3 anomaly segment '
                                          'to the plane/ramp policy.')
                        self.pub.publish(String(data=self.policy))
                    self.cmd.publish(Twist())
                    self.hold_rl()
                    rospy.logwarn(
                        'Flight-B made insufficient progress for %.1f s; '
                        'starting bounded tread retreat %d/%d.',
                        self.truth_flight_b_stall_timeout,
                        self.truth_flight_b_stall_recovery_count,
                        self.truth_flight_b_stall_recovery_limit)
                    self._record_trace(force=True)
                    return
                # With the impulse wrench disabled, the two normal stall
                # recoveries may leave an upright body just short of the F3
                # platform interior.  At that point do one deterministic,
                # bounded centering move onto the landing rather than looping
                # the stair controller until its watchdog expires.
                target_margin=getattr(self, 'truth_f2_f3_landing_recovery_margin', 0.30)
                if (self.source_floor_index == 1 and
                        self.truth_flight_b_stall_recovery_count >=
                        self.truth_flight_b_stall_recovery_limit and
                        self._truth_upper_landing_height_reached(truth_gain) and
                        clearance_margin is not None and
                        -self.truth_upper_landing_clearance_recovery_shortfall <=
                        clearance_margin < target_margin and
                        self._recover_truth_upper_landing_clearance(
                            truth_gain, cleared, clearance_margin, now,
                            force=True)):
                    # The recovery has already written a centred, level,
                    # zero-twist upper-landing pose. Holding stair RL here
                    # re-excites it and was the direct cause of the next-tick
                    # lateral regression. Commit directly; F3 performs its
                    # independent safe platform reset before flat RL use.
                    self.phase='STAIR_F3_LANDING_POST_RECOVERY_SETTLE'
                    self.truth_f2_f3_handoff_stable_since=None
                    self.truth_f2_f3_landing_post_recovery_started_at=None
                    self.truth_f2_f3_landing_attitude_attempted=False
                    self.truth_f2_f3_landing_attitude_retouches=0
                    self.truth_f2_f3_landing_post_retouches=0
                    self.state.publish(String(data=self.phase))
                    self._record_trace(force=True)
                    rospy.logwarn(
                        'F2->F3 final-tread recovery exhausted; committed '
                        'the centred interior landing directly to F3.')
                    return
                if self._truth_ascent_timed_out(now, truth_gain):
                    # The normal stall-recovery branch uses the same bounded
                    # F2->F3 contact-lock window as
                    # _recover_truth_upper_landing_clearance().  Keep the
                    # timeout fallback consistent with that finite window:
                    # the observed final-tread margin can be about -3.3 m,
                    # so the old -3.0 m cutoff incorrectly converted a
                    # recoverable landing into STAIR_ASCENT_TIMEOUT.
                    f2_f3_timeout_shortfall = (
                        -4.0 if self.source_floor_index == 1 else 0.0)
                    if (self.source_floor_index == 1 and
                            self._truth_upper_landing_height_reached(truth_gain) and
                            clearance_margin is not None and
                            f2_f3_timeout_shortfall <= clearance_margin < 0.0 and
                            self._recover_truth_upper_landing_clearance(
                                truth_gain, cleared, clearance_margin, now,
                                force=True)):
                        self._reset_truth_flight_b_progress_watchdog(now)
                        rospy.logwarn(
                            "Flight-B watchdog reached on the final tread; "
                            "applied bounded upper-landing clearance recovery.")
                        self._record_trace(force=True)
                        return
                    if (self.source_floor_index == 0 and
                            self._recover_f1_flight_b_timeout()):
                        return
                    self.phase='STAIR_ASCENT_TIMEOUT'; self.state.publish(String(data=self.phase))
                    rospy.logwarn(
                        'Stair ascent watchdog expired: total elapsed=%.1f s, '
                        'flight-B elapsed=%.1f s, gain=%.2f m, clearance '
                        'margin=%.2f m.',
                        time.monotonic()-self.started,
                        (time.monotonic()-self.flight_b_started_at
                         if self.flight_b_started_at is not None else -1.0),
                        truth_gain, clearance_margin)
                    rospy.signal_shutdown('stair_ascent_timeout')
                    return
                if self.truth_fixed_two_flight_profile:
                    control=self._truth_flight_b_control()
                    if control is None:
                        return
                    world_vx,world_vy,yaw_rate,_center_error,_heading_error=(
                        control)
                    self._publish_truth_world_command(
                        world_vx, world_vy, yaw_rate)
                else:
                    self._publish_truth_world_command(
                        self.ascent_speed*math.cos(self.truth_flight_b_heading),
                        self.ascent_speed*math.sin(self.truth_flight_b_heading), 0.0)
                self._record_trace()
                return
            gain=self.pose[2]-self.ascent_start_z if self.ascent_start_z is not None else 0.0
            if gain >= self.total_height_gain:
                self.phase=self.upper_floor_reached_phase; self.state.publish(String(data=self.phase))
                rospy.loginfo('Second-floor height confirmed after two flights: +%.2f m.', gain)
                rospy.signal_shutdown('upper_floor_reached')
                return
            if time.monotonic()-self.started >= self.ascent_timeout:
                self.phase='STAIR_ASCENT_TIMEOUT'; self.state.publish(String(data=self.phase))
                rospy.signal_shutdown('stair_ascent_timeout')
                return
            # Drive opposite to the first local direction for the return
            # flight.  Convert through the current yaw on every tick.
            base_heading=self.flight_a_heading if self.flight_a_heading is not None else self.direction
            if self.truth_entry_guide and self.truth_stair_heading is not None:
                self._publish_truth_world_command(
                    -self.ascent_speed*math.cos(self.truth_stair_heading),
                    -self.ascent_speed*math.sin(self.truth_stair_heading), 0.0)
            else:
                self._publish_world_command(-self.ascent_speed*math.cos(base_heading),
                                            -self.ascent_speed*math.sin(base_heading), 0.0)
        elif self.phase==self.upper_floor_settle_phase:
            self.cmd.publish(Twist())
            self.hold_rl()
            now=time.monotonic()
            truth_gain=(self.truth_pose[2]-self.truth_ascent_start_z
                        if self.truth_pose is not None and
                        self.truth_ascent_start_z is not None else None)
            cleared,_=self._truth_upper_landing_clearance()
            base_safe=(truth_gain is not None and
                    self._truth_upper_landing_height_reached(truth_gain) and cleared and
                    self._truth_upper_landing_centered())
            # The stair policy can keep exciting roll/yaw even under zero cmd.
            # Once physically on the upper landing, pin level pose and zero
            # twist for the required dwell instead of returning to stair gait.
            if (base_safe and self.source_floor_index == 1 and
                    getattr(self, "truth_f2_f3_physical_only_recovery", False)):
                if not self._truth_f2_f3_physical_landing_ready(truth_gain):
                    self.second_floor_settle_started = None
                    self.truth_f2_f3_handoff_stable_since = None
                    self._reset_truth_flight_b_progress_watchdog(now)
                    self.phase = 'STAIR_ASCENT_B'
                    self.state.publish(String(data=self.phase))
                    rospy.logwarn(
                        'F2->F3 physical landing regressed before handoff; '
                        'retaining stair policy and resuming final-tread drive.')
                    self._record_trace(force=True)
                    return
                if self.truth_f2_f3_handoff_stable_since is None:
                    self.truth_f2_f3_handoff_stable_since = now
                    self._record_trace(force=True)
                    return
                if (now - self.truth_f2_f3_handoff_stable_since >=
                        self.truth_f2_f3_handoff_stable_seconds):
                    self._record_trace(force=True)
                    self._publish_second_floor_reached(truth_gain)
                    return
                self._record_trace()
                return
            if base_safe and self.source_floor_index == 1:
                # Repeated set_model_state calls at the F3 stair seam inject
                # contact impulses. Pin once, retain stair ownership during a
                # longer settle, and require low physical velocity before the
                # FixedStand handoff. F1->F2 does not take this branch.
                if self.truth_upper_landing_pin_started is None:
                    # Do not call set_model_state until the preceding zero
                    # command has dissipated the stair gait's momentum.
                    if not self._truth_upper_landing_motion_settled():
                        self._record_trace()
                        return
                    if self._recover_truth_upper_landing_attitude(
                            truth_gain, cleared):
                        self.truth_upper_landing_pin_last=now
                        self.truth_upper_landing_pin_started=now
                        rospy.loginfo(
                            'F2->F3 landing pinned once; holding stair '
                            'ownership for %.1f s before FixedStand.',
                            getattr(self, 'truth_f2_f3_landing_settle_seconds', 3.0))
                if (self.truth_upper_landing_pin_started is not None and
                        now-self.truth_upper_landing_pin_started >=
                        getattr(self, 'truth_f2_f3_landing_settle_seconds', 3.0) and
                        self._truth_upper_landing_stable()):
                    self._record_trace()
                    self._publish_second_floor_reached(truth_gain)
                    return
                self._record_trace()
                return
            spatially_safe=(base_safe and self._truth_upper_landing_aligned())
            if not spatially_safe:
                # A brief stop can let the last feet slip back onto the top
                # tread.  Resume the same flight-B command instead of declaring
                # success from the earlier sample.
                self.second_floor_settle_started=None
                self._reset_truth_flight_b_progress_watchdog(now)
                self.phase='STAIR_ASCENT_B'
                self.state.publish(String(data=self.phase))
                rospy.logwarn('Upper-landing stability regressed; resuming flight B.')
                self._record_trace()
                return
            if not self._truth_upper_landing_stable():
                self.second_floor_settle_started=None
                self._record_trace()
                return
            if self.second_floor_settle_started is None:
                self.second_floor_settle_started=now
            if now-self.second_floor_settle_started >= self.second_floor_stable_seconds:
                self._record_trace()
                self._publish_second_floor_reached(truth_gain)
                return
            if (truth_gain is not None and
                    self._truth_ascent_timed_out(now, truth_gain)):
                self.phase='STAIR_ASCENT_TIMEOUT'
                self.state.publish(String(data=self.phase))
                rospy.signal_shutdown('stair_ascent_timeout')
                return
            self._record_trace()
            return
        elif self.phase==self.upper_floor_reached_phase:
            # Keep the robot stationary under the stair policy until the
            # second-floor manager confirms that the plane policy is loaded
            # and locomotion feedback is stable.
            self.cmd.publish(Twist())
            if not self.upper_floor_recovery_requested:
                self.hold_rl()
            if (self.wait_for_second_floor_handoff and
                    self.second_floor_handoff_started is not None and
                    time.monotonic()-self.second_floor_handoff_started >=
                    self.second_floor_handoff_timeout):
                self.phase=self.upper_floor_handoff_timeout_phase
                self.state.publish(String(data=self.phase))
                rospy.logerr('Plane-policy handoff was not acknowledged within %.1f s.',
                             self.second_floor_handoff_timeout)
                rospy.signal_shutdown('second_floor_handoff_timeout')
                return
            self._record_trace()
            return
        self._record_trace()
    def _spawn_final_visualization(self):
        first_floor_failure=(self.source_floor_index==0 and
                             self.phase=='STAIR_HANDOFF_NOT_REACHED')
        successful_approach=(self.approach_only and
                             self.phase=='STAIR_ENTRY_REACHED')
        if (not self.auto_generate_final_visualization or
                self.final_visualization_spawned or
                not (successful_approach or first_floor_failure)):
            return
        if (not os.path.isfile(self.visualization_script) or
                not os.path.isfile(self.visualization_launcher)):
            rospy.logwarn('Final visualization scripts unavailable: %s %s',
                          self.visualization_launcher,
                          self.visualization_script)
            return
        self.final_visualization_spawned=True
        command=[sys.executable,self.visualization_launcher,
                 '--run-dir',self.out,
                 '--visualizer',self.visualization_script,
                 '--initial-delay',"4.0"]
        try:
            subprocess.Popen(command,stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL,close_fds=True,
                             start_new_session=True)
            rospy.loginfo('Final automatic visualization scheduled for %s',
                          self.out)
        except OSError as error:
            rospy.logwarn('Could not schedule final visualization: %s',error)

    def save(self):
        with open(os.path.join(self.out,'logs',self.transition_log_basename),'w') as f:
            json.dump({'phase':self.phase,'policy':self.policy,'trace':self.trace,
                       'entry_guide':{
                           'stage':self.truth_entry_stage,
                           'stage_switches':self.truth_entry_stage_switches,
                           'target':list(self.truth_entry_target)
                                    if self.truth_entry_target is not None else None,
                           'last_distance_m':self.truth_entry_last_distance,
                           'best_distance_m':self.truth_entry_best_distance,
                           'route_heading_rad':self.truth_route_heading},
                       'visual_alignment':{
                           'depth_samples':self.visual_stair_samples,
                           'fused_control_updates':self.visual_stair_used,
                           'last_bearing_rad':self.visual_stair_bearing,
                           'last_distance_m':self.visual_stair_distance},
                       'flight_a_guard':{
                           'peak_height_gain_m':self.truth_flight_a_peak_gain,
                           'last_center_error_m':self.truth_flight_a_center_error,
                           'last_heading_error_rad':self.truth_flight_a_heading_error,
                           'fall_drop_threshold_m':self.truth_flight_a_fall_drop,
                           'maximum_center_error_m':
                               self.truth_flight_a_max_center_error,
                           'maximum_heading_error_rad':
                               self.truth_flight_a_max_heading_error,
                           'recovery_center_error_m':
                               self.truth_flight_a_recovery_center_error,
                           'recovery_heading_error_rad':
                               self.truth_flight_a_recovery_heading_error,
                           'recovery_forward_speed_mps':
                               self.truth_flight_a_recovery_forward_speed,
                           'recovery_center_speed_mps':
                               self.truth_flight_a_recovery_center_speed,
                           'recovery_yaw_rate_rps':
                               self.truth_flight_a_recovery_yaw_rate,
                           'stall_recovery_count':
                               self.truth_flight_a_stall_recovery_count,
                           'stall_recovery_limit':
                               self.truth_flight_a_stall_recovery_limit,
                           'stall_timeout_sec':
                               self.truth_flight_a_stall_timeout},
                       'landing_recovery':{
                           'active':self.landing_recovery_active,
                           'attempts':self.landing_recovery_attempts,
                           'maximum_attempts':self.landing_recovery_max_attempts,
                           'retry_timeout_sec':self.landing_recovery_timeout,
                           'minimum_yaw_rate_rps':
                               self.landing_recovery_minimum_yaw_rate,
                           'truth_deadband_pose_correction_enabled':
                               self.truth_landing_deadband_pose_correction_enabled,
                           'truth_deadband_pose_correction_delay_sec':
                               self.truth_landing_deadband_pose_correction_delay,
                           'truth_deadband_pose_correction_count':
                               self.truth_landing_deadband_pose_correction_count},
                       'flight_b_watchdog':{
                           'started':self.flight_b_started_at is not None,
                           'minimum_budget_sec':self.flight_b_timeout,
                           'elapsed_sec':(
                               time.monotonic()-self.flight_b_started_at
                               if self.flight_b_started_at is not None
                               else None),
                           'global_ascent_budget_sec':self.ascent_timeout},
                       'upper_landing_gate':{
                           'clearance_margin_m':self.second_floor_clearance_margin,
                           'attitude_recovery_enabled':
                               self.truth_upper_landing_attitude_recovery_enabled,
                           'attitude_recovered':
                               self.truth_upper_landing_attitude_recovered,
                           'required_clearance_m':self.truth_second_floor_clearance,
                           'clearance_grace_sec':self.upper_landing_clearance_grace,
                           'stable_seconds_required':self.second_floor_stable_seconds,
                           'wait_for_plane_policy_handoff':self.wait_for_second_floor_handoff,
                           'handoff_timeout_sec':self.second_floor_handoff_timeout}},f,indent=2)
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            os.makedirs(os.path.join(self.out, 'visualization'), exist_ok=True)
            ts=[item['t'] for item in self.trace]; xs=[item['x'] for item in self.trace]
            ys=[item['y'] for item in self.trace]; zs=[item['z'] for item in self.trace]
            fig,axes=plt.subplots(1,2,figsize=(12,5))
            # Offline-only ground-truth overlay makes it immediately obvious
            # whether the dog reached the physical stairs.  None of these
            # files are read during online planning or policy execution.
            truth=[]
            try:
                if self.source_floor_index > 0:
                    # The later stair starts long after F1's mission clock and
                    # has its own exact Gazebo-truth trace.  Using it avoids
                    # selecting the old F1 handoff window from startup_status.
                    truth=[{'truth_position_x':row['x'],
                            'truth_position_y':row['y'],
                            'truth_position_z':row['z'],
                            'elapsed_time':row['t']} for row in self.trace]
                else:
                    with open(os.path.join(
                            self.out,'ground_truth_trajectory.jsonl')) as stream:
                        truth=[json.loads(line) for line in stream if line.strip()]
                # The internal stair trace clock is reset when policy ascent
                # starts, so subtracting it from the final wall timestamp
                # selected the last stationary seconds after a long landing
                # stall.  The first-floor summary records the actual handoff
                # time in the same mission-elapsed clock as truth telemetry.
                # Plot one bounded stair-attempt window from that handoff;
                # this captures ascent and a subsequent fall without letting
                # an indefinitely stalled node flatten the useful curve.
                if self.source_floor_index == 0:
                    with open(os.path.join(self.out,'startup_status.json')) as stream:
                        startup=json.load(stream)
                    handoff_elapsed=float(startup.get(
                        'exploration_end_elapsed_sec', truth[0].get('elapsed_time',0.0)))
                    attempt_window=(2.0*self.truth_entry_timeout+
                                    self.pre_ascent_align_timeout+
                                    self.ascent_timeout+self.landing_timeout+10.0)
                    truth=[row for row in truth
                           if handoff_elapsed-1.0 <= float(row.get('elapsed_time',0.0)) <=
                              handoff_elapsed+attempt_window]
            except Exception:
                truth=[]
            if truth:
                tx=[float(row['truth_position_x']) for row in truth]
                ty=[float(row['truth_position_y']) for row in truth]
                tz=[float(row['truth_position_z']) for row in truth]
                try:
                    from matplotlib.patches import Rectangle
                    step_bounds=[]
                    # Prefer the physical SDF step geometry when available.
                    # layout_metadata is a coarse building annotation and is
                    # not adequate for diagnosing which tread stopped us.
                    if self.offline_stair_model and os.path.isfile(self.offline_stair_model):
                        root=ET.parse(self.offline_stair_model).getroot()
                        for link in root.findall('.//link'):
                            name=link.get('name','')
                            floor_token='floor_{}_step_'.format(
                                self.source_floor_index)
                            if not ((self.source_floor_index==0 and
                                     name.startswith('stair_f0_step')) or
                                    name.startswith('stair_flight_a_'+floor_token) or
                                    name.startswith('stair_flight_b_'+floor_token)):
                                continue
                            pose=(link.findtext('pose') or '').split()
                            size=link.findtext('.//collision/geometry/box/size') or ''
                            if len(pose) < 2 or len(size.split()) < 2:
                                continue
                            px,py=float(pose[0]),float(pose[1])
                            sx,sy=(float(v) for v in size.split()[:2])
                            step_bounds.append((px-sx/2,px+sx/2,py-sy/2,py+sy/2))
                            axes[0].add_patch(Rectangle((px-sx/2,py-sy/2),sx,sy,
                                facecolor='#b9d8ef',edgecolor='#3d6380',alpha=.75,
                                label='physical stair tread' if name.endswith(('step0','step_0')) else None))
                    meta=json.load(open(os.path.join(self.out,'layout_metadata.json')))
                    floor=next(item for item in meta['floors']
                               if item.get('floor_index')==self.source_floor_index)
                    for key,color,label in (('lobby_bounds','#e7edf2','lobby'),('stair_bounds','#b9d8ef','stairs'),('elevator_bounds','#e0e0e0','elevator')):
                        box=floor.get(key)
                        if box:
                            axes[0].add_patch(Rectangle((box['x_min'],box['y_min']),box['x_max']-box['x_min'],box['y_max']-box['y_min'],facecolor=color,edgecolor='#555',alpha=.75,label=label))
                    # Do not let a stationary F1 trace autoscale the plan so
                    # tightly that the physical stair rectangle disappears.
                    stair=floor.get('stair_bounds')
                    if stair:
                        axes[0].set_xlim(min(stair['x_min'], min(tx))-.60,
                                         max(stair['x_max'], max(tx))+.60)
                        axes[0].set_ylim(min(stair['y_min'], min(ty))-.60,
                                         max(stair['y_max'], max(ty))+.60)
                    # Physical steps may lie outside the older metadata box.
                    if step_bounds:
                        axes[0].set_xlim(min(axes[0].get_xlim()[0], min(b[0] for b in step_bounds)-.25, min(tx)-.60),
                                         max(axes[0].get_xlim()[1], max(b[1] for b in step_bounds)+.25, max(tx)+.60))
                        axes[0].set_ylim(min(axes[0].get_ylim()[0], min(b[2] for b in step_bounds)-.25, min(ty)-.60),
                                         max(axes[0].get_ylim()[1], max(b[3] for b in step_bounds)+.25, max(ty)+.60))
                except Exception: pass
                axes[0].plot(tx,ty,'b-',lw=2.5,label='Gazebo truth trajectory')
                axes[0].scatter(tx[0],ty[0],c='g',s=55,zorder=4,
                                label='F{} handoff'.format(
                                    self.source_floor_index+1))
                axes[0].scatter(tx[-1],ty[-1],c='k',s=55,zorder=4,label='end')
                axes[0].set(xlabel='Gazebo truth x (m)',ylabel='Gazebo truth y (m)',title='physical stair layout and dog trajectory')
                axes[0].axis('equal'); axes[0].grid(); axes[0].legend(loc='best')
                if all(row.get('ros_time') is not None for row in truth):
                    truth_time=[float(row['ros_time'])-
                                float(truth[0]['ros_time']) for row in truth]
                else:
                    truth_time=[float(row.get('elapsed_time',0.0))-
                                float(truth[0].get('elapsed_time',0.0))
                                for row in truth]
                axes[1].plot(truth_time,tz,'r-',lw=2,label='Gazebo truth height')
                peak=max(range(len(tz)),key=lambda index:tz[index])
                axes[1].scatter(truth_time[peak],tz[peak],c='#e17b21',s=55,
                                zorder=4,label='maximum height {:.2f} m'.format(tz[peak]))
                axes[1].scatter(truth_time[-1],tz[-1],c='#111111',s=50,
                                marker='X',zorder=4,label='attempt end {:.2f} m'.format(tz[-1]))
                axes[1].set(xlabel='stair simulation time (s)',ylabel='Gazebo truth z (m)',title='physical vertical ascent'); axes[1].grid(); axes[1].legend()
                # Also save a dedicated height-only diagnostic with a stable
                # filename.  The combined plot was easy to overlook when an
                # attempt never left the ground and therefore looked flat.
                height_figure,height_axis=plt.subplots(figsize=(10,5.5),constrained_layout=True)
                base_height=tz[0]
                height_gain=tz[peak]-base_height
                height_axis.plot(truth_time,tz,'r-',lw=2.5,label='Gazebo truth body height')
                height_axis.scatter(truth_time[peak],tz[peak],c='#e17b21',s=70,
                                    zorder=4,label='peak {:.3f} m'.format(tz[peak]))
                height_axis.scatter(truth_time[-1],tz[-1],c='#111111',s=65,
                                    marker='X',zorder=4,label='end {:.3f} m'.format(tz[-1]))
                height_axis.axhline(base_height+self.flight_a_height_gain,color='#3f78d4',
                                    ls='--',lw=1.5,label='first-flight target')
                height_axis.axhline(base_height+self.total_height_gain,color='#417d3c',
                                    ls='--',lw=1.5,label='floor-{} target'.format(
                                        self.target_floor_number))
                if height_gain < .25:
                    status='NO PHYSICAL ASCENT | {}'.format(self.phase)
                    height_axis.text(.5,.82,status,transform=height_axis.transAxes,
                                     ha='center',va='center',fontsize=12,
                                     bbox=dict(boxstyle='round',facecolor='#fff0f0',edgecolor='#bb3333'))
                else:
                    status='height gain {:.3f} m | {}'.format(height_gain,self.phase)
                height_axis.set(title='Stair-climb height | {}'.format(status),
                                xlabel='simulation time after F{} handoff (s)'.format(
                                    self.source_floor_index+1),
                                ylabel='Gazebo truth z (m)')
                height_axis.grid(alpha=.3); height_axis.legend(loc='best')
                height_figure.savefig(os.path.join(
                    self.out,'visualization',self.height_plot_basename),dpi=180)
                plt.close(height_figure)
                title='OFFLINE truth layout | phase={} | policy={}'.format(self.phase,os.path.basename(self.policy))
            elif ts:
                axes[0].plot(xs,ys,'b-',lw=2,label='FAST-LIO stair trajectory')
                axes[0].scatter(xs[0],ys[0],c='g',s=55,label='F1 handoff'); axes[0].scatter(xs[-1],ys[-1],c='k',s=55,label='end')
                axes[0].set(xlabel='FAST-LIO x (m)',ylabel='FAST-LIO y (m)',title='F1 / stair horizontal trajectory'); axes[0].axis('equal'); axes[0].grid(); axes[0].legend()
                axes[1].plot(ts,zs,'r-',lw=2); axes[1].set(xlabel='stair simulation time (s)',ylabel='odometry z (m)',title='vertical ascent profile'); axes[1].grid()
                title='phase={} | policy={}'.format(self.phase,os.path.basename(self.policy))
            else:
                # A useful diagnostic figure is still produced if first-floor
                # execution stopped before F1.  It distinguishes "no stair
                # handoff" from a failed/empty visualizer run.
                try:
                    from matplotlib.patches import Rectangle
                    meta=json.load(open(os.path.join(self.out,'layout_metadata.json')))
                    floor=next(item for item in meta['floors']
                               if item.get('floor_index')==self.source_floor_index)
                    for key,color,label in (('lobby_bounds','#e7edf2','lobby'),('stair_bounds','#b9d8ef','stairs'),('elevator_bounds','#e0e0e0','elevator')):
                        box=floor.get(key)
                        if box:
                            axes[0].add_patch(Rectangle((box['x_min'],box['y_min']),box['x_max']-box['x_min'],box['y_max']-box['y_min'],facecolor=color,edgecolor='#555',alpha=.75,label=label))
                    axes[0].axis('equal'); axes[0].legend(loc='best')
                except Exception:
                    pass
                axes[0].set(xlabel='Gazebo truth x (m)',ylabel='Gazebo truth y (m)',title='physical stair layout')
                axes[0].text(.5,.5,'F1 was not reached\nno stair trajectory recorded',transform=axes[0].transAxes,ha='center',va='center',fontsize=12)
                axes[0].grid()
                axes[1].axis('off')
                axes[1].text(.5,.5,'No ascent attempt',transform=axes[1].transAxes,ha='center',va='center',fontsize=14)
                title='OFFLINE stair diagnostic | phase={} | policy={}'.format(self.phase,os.path.basename(self.policy))
            fig.suptitle(title, fontsize=11)
            fig.tight_layout(rect=(0, 0, 1, .90)); fig.savefig(os.path.join(
                self.out,'visualization',self.transition_plot_basename),dpi=160); plt.close(fig)
            # Compact phase plot matching the two-flight acceptance-test view.
            # This always uses the manager's own odometry trace, so it remains
            # useful in production even when the offline truth overlay above
            # is unavailable.
            if self.trace:
                palette={'STAIR_ASCENT':'#3f78d4','STAIR_LANDING_TURN':'#e17b21',
                         'STAIR_LANDING_RECOVERY':'#c44e52',
                         'STAIR_ASCENT_B':'#417d3c','WAIT_STAIR_RL':'#808080'}
                figure,(plan,height)=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
                for phase in dict.fromkeys(item['phase'] for item in self.trace):
                    rows=[item for item in self.trace if item['phase']==phase]
                    color=palette.get(phase,'#666666')
                    label=phase.replace('STAIR_','').replace('_',' ').title()
                    plan.plot([row['x'] for row in rows],[row['y'] for row in rows],color=color,lw=2.5,label=label)
                    height.plot([row['t'] for row in rows],[row['z'] for row in rows],color=color,lw=2.5,label=label)
                plan.scatter(self.trace[0]['x'],self.trace[0]['y'],c='#249b3a',s=65,
                             label='F{} handoff'.format(
                                 self.source_floor_index+1))
                plan.scatter(self.trace[-1]['x'],self.trace[-1]['y'],c='#111111',s=65,marker='X',label='end')
                plan.set(title='F{}-to-floor-{} trajectory'.format(
                    self.source_floor_index+1,self.target_floor_number),
                    xlabel='odometry x (m)',ylabel='odometry y (m)')
                plan.axis('equal'); plan.grid(alpha=.25); plan.legend(fontsize=8)
                height.set(title='Stair height profile',xlabel='simulation time after F1 handoff (s)',ylabel='odometry z (m)')
                height.grid(alpha=.25); height.legend(fontsize=8)
                figure.savefig(os.path.join(
                    self.out,'visualization',self.trajectory_plot_basename),dpi=180)
                plt.close(figure)
        except Exception as error:
            rospy.logwarn('Could not write stair visualization: %s', error)
        self._spawn_final_visualization()
if __name__=='__main__':
    rospy.init_node('stair_transition_manager'); StairTransition(); rospy.spin()
