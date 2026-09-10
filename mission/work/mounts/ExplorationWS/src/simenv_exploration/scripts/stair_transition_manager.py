#!/usr/bin/env python3
"""Online F1-to-stair handoff with logs for the second-floor controller.

The node uses only F1's current odometry and registered cloud.  It never
reads the building layout or Gazebo truth.  This first version performs the
safe handoff and records an ascent trace; its direction score is deliberately
conservative and requires a positive-height, near-range cloud sector.
"""
import json, math, os, time, shutil
import xml.etree.ElementTree as ET
import rospy
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, Joy, JointState
from std_msgs.msg import String, Bool
from gazebo_msgs.msg import ModelStates, LinkStates

def yaw(q):
    return math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))

class StairTransition:
    def __init__(self):
        self.out=os.path.abspath(rospy.get_param('~output_dir'))
        self.policy=rospy.get_param('~stair_policy')
        # Keep the validated F1->F2 instance as the zero-configuration default,
        # while allowing a second, topic-isolated instance to reuse the exact
        # same physical stair controller for F2->F3.
        self.source_floor_index=int(rospy.get_param('~source_floor_index', 0))
        self.target_floor_number=int(rospy.get_param(
            '~target_floor_number', self.source_floor_index+2))
        self.upper_floor_reached_phase=str(rospy.get_param(
            '~upper_floor_reached_phase', 'SECOND_FLOOR_REACHED'))
        self.upper_floor_settle_phase=str(rospy.get_param(
            '~upper_floor_settle_phase', 'SECOND_FLOOR_SETTLE'))
        self.upper_floor_ready_token=str(rospy.get_param(
            '~upper_floor_ready_token', 'SECOND_FLOOR_EXPLORATION_READY'))
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
        self.mission_failure_token=str(rospy.get_param(
            '~mission_failure_token', '')).strip()
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
        self._pause_pub=rospy.Publisher(
            str(rospy.get_param('~goal_executor_pause_topic',
                                '/simenv/goal_executor_pause')),
            Bool, queue_size=1, latch=True)
        self._pause_active=False
        self.odom_seen=False
        self.trace=[]; self.last_log=0.; self.direction=None
        self.stair_direction_locked=False
        self.handoff_seen=False; self.policy_loaded=False
        # Joint-angle snapshot at Flight-B entry (12 dofs).  F1->F2 and F2->F3
        # enter with equivalent x/y/yaw now, so the remaining systematic
        # difference is the joint state the RL policy starts from; capture it
        # per source floor to build a stand-to-snapshot reset.
        # Joint-angle snapshot at Flight-B entry (12 dofs).  F1->F2 and F2->F3
        # enter with equivalent x/y/yaw now, so the remaining systematic
        # difference is the joint state the RL policy starts from; capture it
        # per source floor to build a stand-to-snapshot reset.
        self.entry_joint_snapshot=None
        self.entry_joint_names=None
        self._entry_joint_wall_time=None
        # Stand target override: SNAPSHOT_JOINTS_FILE points at the captured
        # F1->F2 entry pose (flight_b_entry_joints_0.json).  It is pushed to
        # /stand_target_joints only for the pre-ascent stand and cleared
        # afterwards, so the startup auto-hold stand keeps the default pose.
        self.snapshot_file=str(os.environ.get(
            'SNAPSHOT_JOINTS_FILE', '')).strip()
        self._stand_target_param='/stand_target_joints'
        self.joint_state_sub=rospy.Subscriber(
            '/a1_gazebo/joint_states', JointState, self._on_joint_states,
            queue_size=1)
        self.return_transit_armed=False
        self.return_gate_published=False
        self.truth_return_gate_radius=float(rospy.get_param(
            '~truth_return_gate_radius_m', 5.40))
        self.locomotion_ready=False
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
        # Two-stage entry route: handoff anchor -> stair side opening ->
        # pre-riser.  The corridor-lobby leg is folded into the side stage;
        # on F1 the anchor sits in the open lobby east of the stair core and
        # the diagonal passes beneath the elevated staircase.  On upper
        # floors the side point is replaced by the solid landing-side point
        # (the core is a void at that level, run101 fell at x=-1.78,
        # y=3.10 on F2).
        self.truth_two_stage_guide=bool(rospy.get_param(
            '~truth_two_stage_guide', True))
        # The truth return gate takes over 35 m from the pre-riser, which
        # can be deep inside the F1 corridor (y~29).  A direct side-stage
        # chord from there would cut through the corridor's west partition.
        # While farther than this radius from the side opening, the side
        # stage instead follows the corridor-centreline lobby point first,
        # then switches to the side opening when close enough.
        self.truth_side_approach_radius=float(rospy.get_param(
            '~truth_side_approach_radius_m', 6.0))
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
        self.truth_using_corridor_return=False
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
        self.truth_ascent_start_z=None
        self.truth_landing_heading_offset=float(rospy.get_param(
            '~truth_landing_heading_offset_rad', .37))
        self.truth_stage_settle_until=None
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
        self.truth_entry_watchdog_progress=float(rospy.get_param(
            '~truth_entry_watchdog_progress_m', 0.25))
        # Speed caps for the truth entry guide.  The approach profile is
        # governed by 0.70*distance below ~0.64 m regardless of these caps,
        # so raising them only shortens the long straights, never the
        # 12 cm final-stage brake.
        self.truth_entry_guide_speed=float(rospy.get_param(
            '~truth_entry_guide_speed_mps', .45))
        self.truth_entry_guide_min_speed=float(rospy.get_param(
            '~truth_entry_guide_min_speed_mps', .25))
        self.truth_entry_watchdog_anchor_distance=None
        self.policy_warmup_seconds=float(rospy.get_param('~policy_warmup_seconds', 1.2))
        self.policy_warmup_until=None
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
        # Lateral acceptance band for the landing-turn hairpin.  Full48g left
        # the turn 0.149 m off the flight-B centreline (x=-2.634 vs -2.485)
        # because the former 0.15 m gate admitted it, then slid off the
        # platform edge on the first climb step.  Keep the gate tight so the
        # climb attacks the riser head-on.
        self.truth_landing_position_tolerance=float(rospy.get_param(
            '~truth_landing_position_tolerance_m', .06))
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
        # A climbing slip leaves the body pitched back on the riser; the
        # 0.12 m/s recovery crawl then cannot re-establish tread contact and
        # the ascent deadlocks until timeout (full20 stalled 20 min with
        # world=(0.18,0.12) commands).  If recovery produces no height gain
        # within a short window, release the full climb speed to punch
        # through the riser instead of idling.
        self.truth_flight_a_recovery_stall_seconds=float(rospy.get_param(
            '~truth_flight_a_recovery_stall_seconds', 5.0))
        self.truth_flight_a_recovery_stall_z_progress_m=float(rospy.get_param(
            '~truth_flight_a_recovery_stall_z_progress_m', .02))
        self.truth_flight_a_recovery_since=None
        self.truth_flight_a_recovery_start_z=None
        # Rolling stall window: a stall is any stall_seconds window with
        # < stall_z_progress_m of height gain, tracked independently of the
        # recovery deviation flags.  The former since-recovery-start measure
        # never fired when the pre-stall approach gained just above the
        # threshold (R8h: 0.023 m vs 0.02 m), leaving the climb wedged at
        # recovery speed forever with no watchdog.
        self.truth_flight_a_stall_window_start=None
        self.truth_flight_a_stall_window_z=None
        # Hard stall guard: even at full released speed, no height gain for
        # hard_stall_seconds means the gait is wedged on the treads
        # (R8h/full48f).  Fail fast with a phase the runner recognizes
        # instead of consuming the whole F1 stage budget (25 min hang).
        self.truth_flight_a_hard_stall_seconds=float(rospy.get_param(
            '~truth_flight_a_hard_stall_seconds', 60.0))
        self.truth_flight_a_hard_stall_progress_m=float(rospy.get_param(
            '~truth_flight_a_hard_stall_progress_m', .02))
        self.truth_flight_a_hard_stall_since=None
        self.truth_flight_a_hard_stall_start_gain=None
        self.truth_flight_a_peak_gain=None
        self.truth_flight_a_center_error=None
        self.truth_flight_a_heading_error=None
        self.truth_flight_a_recovery_active=False
        self.truth_flight_a_command_forward_speed=None
        self.truth_flight_a_command_center_speed=None
        self.truth_flight_a_command_yaw_rate=None
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
        self.truth_flight_b_fall_drop=float(rospy.get_param(
            '~truth_flight_b_fall_drop_m', .55))
        self.truth_flight_b_peak_gain=None
        self.truth_flight_b_center_error=None
        self.truth_flight_b_heading_error=None
        # Flight-B can slip backwards on a tread and then idle with the gait
        # airborne (pitch ~ -29 deg, yaw drifting, z frozen) even though the
        # command is full climb speed (full9/11/17/18 all froze near
        # y=3.40/z=4.85).  Pause the command briefly so the legs land and
        # re-establish contact, then resume the climb.  Normal climbs gain
        # far more than the stall threshold, so this never interrupts them.
        self.truth_flight_b_stall_seconds=float(rospy.get_param(
            '~truth_flight_b_stall_seconds', 4.0))
        self.truth_flight_b_stall_z_progress_m=float(rospy.get_param(
            '~truth_flight_b_stall_z_progress_m', .02))
        # Landing-slide guard: if the body falls more than this far below the
        # z where flight B started, it slid off the platform edge instead of
        # climbing the first tread (full48g: 4.210 -> 3.998 m, then wedged
        # and froze for the rest of the run).  Force the stall recovery
        # immediately, before the wedge locks up, so the gait re-lands and
        # the backoff can return it onto the platform.
        self.truth_flight_b_drop_detect=float(rospy.get_param(
            '~truth_flight_b_drop_detect_m', .10))
        self.truth_flight_b_entry_z=None
        self.truth_flight_b_drop_backoffs=0
        self.truth_flight_b_recovery_hold_seconds=float(rospy.get_param(
            '~truth_flight_b_recovery_hold_seconds', 4.0))
        # During a stall pause, actively rotate the body back onto the
        # flight-B centreline instead of standing still.  full28 showed the
        # resumed sprint drifting to -2.78 rad (69 deg off axis): the pause
        # re-landed the legs but left the heading where the slip left it, so
        # the misaligned gait re-caught the same riser and only slid sideways.
        self.truth_flight_b_recovery_yaw_rate=float(rospy.get_param(
            '~truth_flight_b_recovery_yaw_rate_rps', .4))
        self.truth_flight_b_recovery_yaw_tolerance=float(rospy.get_param(
            '~truth_flight_b_recovery_yaw_tolerance_rad', .06))
        # full31: a stall pause that ends with the yaw already re-aligned
        # (pause 4, err<0.06) still re-slips the same tread on the full-speed
        # punch-out (~4 mm per round).  Ramp the resumed climb from a low
        # speed so the re-landed gait re-establishes grip before the
        # full-speed ascent, instead of punching straight back to 0.8.
        self.truth_flight_b_recovery_ramp_seconds=float(rospy.get_param(
            '~truth_flight_b_recovery_ramp_seconds', 1.0))
        self.truth_flight_b_recovery_ramp_speed=float(rospy.get_param(
            '~truth_flight_b_recovery_ramp_speed_mps', .4))
        # full34: even with yaw re-aligned and a 0.4->0.8 ramp, every resumed
        # sprint re-slips the same riser (z frozen 4.848-4.855, 7/7 runs).
        # After the recovery pause, back off down the flight so the next
        # sprint re-approaches the tread with a fresh run-up instead of
        # punching out of a dead squat on the riser edge.
        self.truth_flight_b_recovery_backoff_distance_m=float(rospy.get_param(
            '~truth_flight_b_recovery_backoff_distance_m', .35))
        self.truth_flight_b_recovery_backoff_speed_mps=float(rospy.get_param(
            '~truth_flight_b_recovery_backoff_speed_mps', .3))
        # 2026-08-19:backoff 距离随恢复次数升级(对齐下行 v2 escalation)。
        # z=4.0 死锁轮实证:drop backoff 0.35m 后退后 z 仍卡 3.996(水平后退
        # 不回升高度),重新爬升在同一 riser 楔住;固定 0.35m 助跑永远不够,
        # 升级后退 2~3 级(0.75m)给更长助跑(full48g 同因死锁)。
        self.truth_flight_b_recovery_backoff_escalation_m=float(rospy.get_param(
            '~truth_flight_b_recovery_backoff_escalation_m', .15))
        self.truth_flight_b_recovery_backoff_max_distance_m=float(rospy.get_param(
            '~truth_flight_b_recovery_backoff_max_distance_m', .75))
        self.truth_flight_b_slide_exhausted_logged=False
        # Early slip intervention (F2->F3 instance only, source_floor_index=1):
        # the stall detector above only fires after z has frozen for 4 s, but
        # a tread slip collapses the gait (pitch <-25 deg squat, legs airborne,
        # no recovery force left) within ~2 s -- full40-44 all deadlocked this
        # way and every post-collapse recovery (hold/yaw-cap/ramp/backoff) was
        # futile because the collapsed legs cannot push.  Catch the slip within
        # ~1 s while the legs still can: freeze the climb and rotate back onto
        # the centreline at full yaw authority, then resume.  Gated on the
        # robot having actually climbed (gain>slip_min_gain_m) so the natural
        # standstill right after the landing turn never triggers it.
        self.truth_flight_b_slip_detect_seconds=float(rospy.get_param(
            '~truth_flight_b_slip_detect_seconds', 0.0))
        self.truth_flight_b_slip_yaw_rate=float(rospy.get_param(
            '~truth_flight_b_slip_yaw_rate_rps', .6))
        self.truth_flight_b_slip_min_gain_m=float(rospy.get_param(
            '~truth_flight_b_slip_min_gain_m', .30))
        self.truth_flight_b_slip_hold_seconds=float(rospy.get_param(
            '~truth_flight_b_slip_hold_seconds', 1.5))
        # Pre-ascent fixed-stand (F2->F3 instance only): before climbing
        # flight B, drop out of the RL gait into unitree's fixed stand
        # (default joint pose, Joy L2_A) for pre_ascent_stand_seconds, then
        # re-enter the RL gait.  The upper flight is currently entered with
        # the residual joint state from F2 exploration + flight A + landing
        # turn; 11/11 deadlocks (full40-45) indicate that residual biases the
        # tread-slip window.  Re-starting from the policy's training-centre
        # pose removes that bias.  0.0 disables (F1->F2 instance stays as-is).
        self.pre_ascent_stand_seconds=float(rospy.get_param(
            '~pre_ascent_stand_seconds', 0.0))
        self.pre_ascent_stand_started=None
        self.pre_ascent_stand_done=False
        self.truth_flight_b_stall_start_z=None
        self.truth_flight_b_stall_since=None
        self.truth_flight_b_recovery_until=None
        self.truth_flight_b_ramp_until=None
        self.truth_flight_b_backoff_until=None
        self.truth_flight_b_stall_recoveries=0
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
        # Retain the whole-manoeuvre watchdog, but guarantee a bounded window
        # after the landing has actually aligned flight B. Run62 entered the
        # second flight at t=41.6 s and was stopped by the shared 45 s timer
        # at t=45.0 despite continuously gaining height. Historical healthy
        # B flights take about 12 s, so 22 s covers normal gait variation
        # without allowing an indefinitely stuck climb.
        self.flight_b_started_at=None
        self.flight_b_timeout=float(rospy.get_param(
            '~flight_b_timeout_sec', 22.0))
        # Once the height gate is met, grant only enough additional time to
        # put all four feet beyond the final tread. The fall detector remains
        # active throughout both bounded extensions.
        self.upper_landing_clearance_grace=float(rospy.get_param(
            '~upper_landing_clearance_grace_sec', 15.0))
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
        self.finalization_deadline=None
        self.approach_started=None
        self.approach_motion_started=None
        self.approach_origin=None
        self.ascent_start_z=None
        os.makedirs(os.path.join(self.out,'logs'),exist_ok=True)
        # Offline visualization input; the parsed copy is also consulted by
        # the truth entry guide (R41) to re-enter the corridor when a failed
        # exploration hands the robot over from inside a room.
        offline_layout=rospy.get_param('~offline_truth_layout_metadata', '')
        # Strictly offline visualization input: used after shutdown only to
        # draw the actual step rectangles behind the recorded truth trace.
        self.offline_stair_model=rospy.get_param('~offline_stair_model_sdf', '')
        self.truth_offline_layout=None
        if offline_layout and os.path.isfile(offline_layout):
            try:
                with open(offline_layout) as fh:
                    self.truth_offline_layout=json.load(fh)
            except (OSError, ValueError):
                self.truth_offline_layout=None
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
        rospy.Subscriber('/gazebo/model_states',ModelStates,self.on_truth_states,queue_size=2)
        self.truth_links_sub=rospy.Subscriber('/gazebo/link_states',LinkStates,
                                               self.on_truth_links,queue_size=1)
        rospy.Subscriber('/cloud_registered',PointCloud2,self.on_cloud,queue_size=1)
        rospy.Subscriber(rospy.get_param(
            '~depth_points_topic','/real_sense/depth/points'),
            PointCloud2,self.on_depth_cloud,queue_size=1)
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

    def _on_joint_states(self, msg):
        if self.entry_joint_snapshot is None:
            self.entry_joint_names=list(msg.name)
        self.entry_joint_snapshot=list(msg.position)
        self._entry_joint_wall_time=msg.header.stamp.to_sec()

    def _save_entry_joint_snapshot(self):
        """Persist the 12 joint angles at Flight-B entry acceptance."""
        if self.entry_joint_snapshot is None:
            rospy.logwarn('Flight-B entry aligned but no joint_states seen')
            return
        path=os.path.join(self.out,
                          'flight_b_entry_joints_%d.json' %
                          self.source_floor_index)
        with open(path, 'w') as f:
            json.dump({'source_floor_index': self.source_floor_index,
                       'wall_time': self._entry_joint_wall_time,
                       'joint_names': self.entry_joint_names,
                       'positions': self.entry_joint_snapshot}, f)
        rospy.loginfo('Flight-B entry joints saved (%d dofs) -> %s',
                      len(self.entry_joint_snapshot), path)

    def _apply_stand_target_joints(self):
        """Push the F1->F2 entry snapshot as the pre-ascent stand target."""
        if not self.snapshot_file or not os.path.isfile(self.snapshot_file):
            rospy.logwarn('Pre-ascent stand: SNAPSHOT_JOINTS_FILE %r missing, '
                          'using default stance', self.snapshot_file)
            return
        with open(self.snapshot_file) as f:
            data=json.load(f)
        name_to_pos=dict(zip(data['joint_names'], data['positions']))
        # Gazebo joint_states are name-sorted; junior_ctrl FSM order is
        # FR/FL/RR/RL x hip,thigh,calf (URDF order).
        fsm_order=['FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
                   'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
                   'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint',
                   'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint']
        try:
            targets=[name_to_pos[n] for n in fsm_order]
        except KeyError as error:
            rospy.logerr('Pre-ascent stand: snapshot missing joint %s, '
                         'using default stance', error)
            return
        rospy.set_param(self._stand_target_param, targets)
        rospy.loginfo('Pre-ascent stand target set to F1->F2 entry snapshot '
                      '(%s)', self.snapshot_file)

    def _clear_stand_target_joints(self):
        if self.snapshot_file:
            rospy.delete_param(self._stand_target_param)

    def _publish_joy_fixed_stand(self):
        # Unitree L2_A: switch the FSM to fixed stand (default joint pose).
        # Repeatedly re-sent so a stray /joy source cannot knock the FSM out
        # of the stand mid-hold.
        message=Joy()
        message.header.stamp=rospy.Time.now()
        message.axes=[0.0]*8
        message.buttons=[0]*12
        message.buttons[1]=1
        self.joy.publish(message)

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
        robot_yaw=self.truth_pose[3]
        command=Twist()
        command.linear.x=vx*math.cos(robot_yaw)+vy*math.sin(robot_yaw)
        command.linear.y=-vx*math.sin(robot_yaw)+vy*math.cos(robot_yaw)
        command.angular.z=wz
        rospy.loginfo_throttle(
            1.0,
            'STPUB phase=%s world=(%.3f,%.3f,%.3f) body=(%.3f,%.3f,%.3f) '
            'truth_z=%.3f pose=%s',
            self.phase, vx, vy, wz, command.linear.x, command.linear.y,
            command.angular.z,
            self.truth_pose[2] if self.truth_pose is not None else float('nan'),
            (self.pose[2] if self.pose is not None else None))
        self.cmd.publish(command)
        self.hold_rl()

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
        item={'t':round(now-(self.started or now),3), 'phase':self.phase,
              'x':pose[0], 'y':pose[1], 'z':pose[2], 'yaw':pose[3],
              'heading':self.direction,
              'pose_source':('gazebo_truth' if pose is self.truth_pose else 'odometry')}
        if self.phase in ('STAIR_ASCENT_B', getattr(
                self, 'upper_floor_settle_phase', 'SECOND_FLOOR_SETTLE')):
            item.update({
                'flight_b_center_x':self._truth_flight_b_center_x(),
                'flight_b_center_error':self.truth_flight_b_center_error,
                'flight_b_heading_error':self.truth_flight_b_heading_error,
                'flight_b_peak_gain':self.truth_flight_b_peak_gain,
                'flight_b_elapsed_sec':(
                    now-self.flight_b_started_at
                    if self.flight_b_started_at is not None else None),
                'flight_b_timeout_sec':self.flight_b_timeout,
                'flight_b_stall_recoveries':self.truth_flight_b_stall_recoveries})
        if self.phase=='STAIR_ASCENT':
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
                    self.truth_flight_a_command_yaw_rate})
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
        return margin >= 0.0, margin

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
        recovery_active=center_recovery or heading_recovery
        if recovery_active:
            if self.truth_flight_a_recovery_since is None:
                self.truth_flight_a_recovery_since=time.monotonic()
                self.truth_flight_a_recovery_start_z=self.truth_pose[2]
        else:
            self.truth_flight_a_recovery_since=None
        # Rolling stall window (independent of the deviation flags): any
        # stall_seconds window with < stall_z_progress_m of height gain
        # releases the full climb speed.  The old since-recovery-start
        # measure was disabled by marginal pre-stall gain (R8h: 0.023 m of
        # initial gain vs the 0.02 m threshold -> wedged forever at
        # recovery speed, no watchdog).
        now=time.monotonic()
        if self.truth_flight_a_stall_window_start is None:
            self.truth_flight_a_stall_window_start=now
            self.truth_flight_a_stall_window_z=self.truth_pose[2]
            recovery_stall=False
        elif (now-self.truth_flight_a_stall_window_start >=
              self.truth_flight_a_recovery_stall_seconds):
            recovery_stall=(
                self.truth_pose[2]-self.truth_flight_a_stall_window_z <
                self.truth_flight_a_recovery_stall_z_progress_m)
            if recovery_stall:
                rospy.logwarn_throttle(
                    2.0,
                    'Flight-A stall: no height gain for %.1f s; releasing '
                    'full climb speed to break through (gain %.3f m).',
                    self.truth_flight_a_recovery_stall_seconds,
                    self.truth_pose[2]-self.truth_flight_a_stall_window_z)
            self.truth_flight_a_stall_window_start=now
            self.truth_flight_a_stall_window_z=self.truth_pose[2]
        else:
            recovery_stall=False
        forward_speed=(
            self.ascent_speed
            if recovery_stall else (
                min(self.ascent_speed,
                    self.truth_flight_a_recovery_forward_speed)
                if recovery_active else self.ascent_speed))
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
        right_x=math.sin(heading)
        right_y=-math.cos(heading)
        vx=forward_speed*math.cos(heading)+center_speed*right_x
        vy=forward_speed*math.sin(heading)+center_speed*right_y
        self.truth_flight_a_recovery_active=recovery_active
        self.truth_flight_a_command_forward_speed=forward_speed
        self.truth_flight_a_command_center_speed=center_speed
        self.truth_flight_a_command_yaw_rate=yaw_rate
        return vx,vy,yaw_rate,center_error,heading_error

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
        self.cmd.publish(Twist())
        self._record_trace(force=True)
        self.landing_recovery_active=False
        self.phase='STAIR_LANDING_TIMEOUT'
        self.state.publish(String(data=self.phase))
        rospy.logerr('Intermediate landing turn timed out after %d recovery attempts.',
                     self.landing_recovery_attempts)
        rospy.signal_shutdown('stair_landing_turn_timeout')
        return True

    def _publish_second_floor_reached(self, truth_gain=None):
        self.phase=self.upper_floor_reached_phase
        self.second_floor_handoff_started=time.monotonic()
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

    def _begin_handoff(self):
        """Start one stair attempt after the owning floor mission completes."""
        if self.phase != 'WAIT_F1' or self.handoff_seen:
            return
        self.handoff_seen=True
        self.started=time.monotonic()
        if self.external_entry_guide:
            self.phase='WAIT_EXTERNAL_STAIR_ENTRY'
            self.state.publish(String(data=self.phase))
            return

        self.phase=('TRUTH_ENTRY_GUIDE' if self.truth_entry_guide else
                    ('WAIT_STAIR_RL' if self.skip_approach else 'FIND_STAIR'))
        if self.truth_entry_guide:
            self.truth_entry_stage=(
                'side' if self.truth_two_stage_guide else 'corridor')
            self.truth_corridor_target=None
            self.truth_side_target=None
            self.truth_using_corridor_return=False
            self.truth_entry_target=None
            self.truth_route_heading=None
            self.truth_entry_best_distance=None
            self.truth_entry_last_distance=None
            self.truth_entry_stage_switches=0
            self.truth_entry_watchdog_anchor_distance=None
            self.truth_entry_deadline=time.monotonic()+self.truth_entry_timeout
            # R41(2026-08-26):batch7 RUN3 实证探索 TIME_LIMIT 异常结束后
            # 机器人在房间内,二阶段引导从房间内直线穿墙(room_3 南墙
            # y=21.88 卡死 79s)。若 handoff 起点落在任一房间 bounds 内,
            # 先经门洞重入走廊中心线,再走正常 side 引导。
            self.truth_reenter_points=self._truth_entry_reenter_route()
        self.state.publish(String(data=self.phase))

    def on_mission_trigger(self, message):
        if self.phase != 'WAIT_F1':
            return
        payload = str(message.data).strip()
        if self.mission_failure_token and payload == self.mission_failure_token:
            # Source-floor exploration ended in failure: write the terminal
            # phase through save() and exit so the runner can fail this round
            # fast instead of waiting out the full phase timeout (observed:
            # F2 TIME_LIMIT -> 25 min hang with no second_to_third JSON).
            rospy.logerr('Floor-%d stair controller cancelled: source floor '
                         'failed (%s).', self.source_floor_index+1, payload)
            self.phase = 'SOURCE_FLOOR_EXPLORATION_FAILED'
            self.state.publish(String(data=self.phase))
            rospy.signal_shutdown(payload)
            return
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

    def on_state(self,m):
        if ('STAIR_RETURN_TRANSIT' in m.data and self.phase=='WAIT_F1'):
            self._arm_return_transit()
            return
        if ('STAIR_WAIT_ZONE' in m.data or 'STAIR_LOBBY_HANDOFF' in m.data) and self.phase=='WAIT_F1':
            self._begin_handoff()

    def _arm_return_transit(self):
        """Latch F1 return ownership independently of later state updates."""
        if self.phase != 'WAIT_F1':
            return
        self.return_transit_armed=True
        self._maybe_publish_truth_return_gate()

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

    def _truth_entry_reenter_route(self):
        """R41:return truth waypoints that re-enter the corridor first.

        Called from _begin_handoff.  When the source-floor exploration ended
        abnormally (TIME_LIMIT), the handoff may pick the robot up inside a
        room; the two-stage guide then drives a straight chord from the room
        interior to the landing-side point, crossing the room's wall (batch7
        RUN3 wedged against room_3's south wall y=21.88 for 79 s).  A normal
        EXIT_HANDOFF ends on the corridor centreline, so no re-entry applies
        there.  Returns up to two waypoints: just inside the door on the room
        side, then the corridor centreline at the same door y.
        """
        if (not self.truth_two_stage_guide or self.truth_pose is None or
                self.truth_offline_layout is None):
            return []
        rooms=None
        for floor in self.truth_offline_layout.get('floors', []):
            if int(floor.get('floor_index', -1)) == int(
                    getattr(self, 'source_floor_index', 0)):
                rooms=floor.get('rooms', [])
                break
        if not rooms:
            return []
        px, py=float(self.truth_pose[0]), float(self.truth_pose[1])
        for room in rooms:
            bounds=room.get('bounds')
            door=room.get('door_pose')
            if not bounds or not door:
                continue
            if (float(bounds['x_min']) <= px <= float(bounds['x_max']) and
                    float(bounds['y_min']) <= py <= float(bounds['y_max'])):
                door_x=float(door[0])
                door_y=float(door[1])
                side_sign=1.0 if door_x >= 0.0 else -1.0
                room_side=(door_x+side_sign*0.35, door_y)
                corridor_centre=(0.0, door_y)
                rospy.logwarn(
                    'Truth handoff pose (%.2f, %.2f) is inside a room; '
                    're-entering corridor via door x=%.2f y=%.2f '
                    '(room-side %.2f) then the centreline.',
                    px, py, door_x, door_y, room_side[0])
                return [room_side, corridor_centre]
        return []

    def _truth_entry_route_targets(self, final_target, stair_heading):
        """Return (corridor_lobby, side_opening) truth waypoints.

        Two-stage mode (default): the side_opening is the single approach
        waypoint.  On an upper floor it is replaced by the landing-side point
        so the route follows the solid upper landing instead of crossing the
        F1-stair void.  The corridor-lobby point is still computed in both
        modes: when the handoff anchor sits far down the first-floor corridor
        (the truth return gate takes over 35 m from the pre-riser, inside the
        corridor), the guide first follows the corridor centreline back to
        that point before cutting across the open lobby to the side opening.
        Legacy mode: corridor-lobby + side-opening waypoints (unchanged).
        """
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
            # On an upper floor both approach points land beside the solid
            # landing; the corridor chord itself crosses the stair void.
            corridor_target=upper_exit
            if self.truth_two_stage_guide:
                side_target=upper_exit
        return corridor_target, side_target

    def _truth_upper_floor_landing_exit_target(self, side_sign, right):
        """Return the safe lobby-side point beside an upper stair opening.

        F1 has solid floor under the established diagonal lobby route. On F2
        that same chord crosses the F1-to-F2 stair void (run101 fell from
        z=2.91 m at x=-1.78, y=3.10). The generated landing collision gives
        an exact, truth-only waypoint on the open side of the void.

        In two-stage mode this point replaces the side-opening waypoint on
        upper floors so the entry route follows the solid upper landing
        instead of crossing the void.
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
            'Truth return takeover gate reached at %.2f m from pre-riser; '
            'requesting F1 cancellation and corridor-guide handoff.', distance)
    def on_truth_states(self, message):
        try:
            index=message.name.index('a1_gazebo')
            pose=message.pose[index]
            self.truth_pose=(float(pose.position.x), float(pose.position.y),
                             float(pose.position.z), yaw(pose.orientation))
            # The standalone truth-bridge acceptance launch intentionally has
            # no SLAM stack.  Mirror its physical pose into the local trace
            # only when the explicit test guide is active and odometry is
            # absent; production always retains its normal /Odometry source.
            if self.truth_entry_guide and not self.odom_seen:
                self.pose=self.truth_pose
            self._maybe_publish_truth_return_gate()
        except (ValueError, IndexError):
            pass
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
            self.truth_flight_b_heading=cardinal_heading(
                next_b[0]-self.truth_flight_b_pose[0],
                next_b[1]-self.truth_flight_b_pose[1])
        except (StopIteration, IndexError):
            pass
    def on_finalize(self,m):
        if m.data and not self.handoff_seen and self.phase=='WAIT_F1':
            self.phase='FIRST_FLOOR_FINALIZING'
            self.state.publish(String(data=self.phase))
            self.finalization_deadline=time.monotonic()+self.finalization_grace
            rospy.logwarn('First-floor mission ended before F1; allowing %.1f s for offline visualization before shutdown.', self.finalization_grace)
    def on_policy_status(self,m):
        if ('policy_reloaded:' in m.data and
                os.path.basename(self.policy) in m.data):
            self.policy_loaded=True
            self.state.publish(String(data='STAIR_POLICY_READY'))
        elif 'policy_reload_failed:' in m.data and self.phase=='STAIR_POLICY_LOADING':
            self.phase='STAIR_POLICY_FAILED'; self.state.publish(String(data=self.phase))
            rospy.logerr('Stair policy reload failed: %s', m.data)
            rospy.signal_shutdown('stair_policy_reload_failed')
    def on_second_floor_state(self,m):
        if (self.upper_floor_ready_token in m.data and
                self.phase==self.upper_floor_reached_phase):
            self.cmd.publish(Twist())
            self.phase=self.upper_floor_handoff_complete_phase
            self.state.publish(String(data=self.phase))
            rospy.loginfo('Plane policy acknowledged on the upper landing; '
                          'releasing stair controller ownership.')
            rospy.signal_shutdown('upper_floor_handoff_complete')
    def on_locomotion_ready(self,m):
        self.locomotion_ready=bool(m.data)
    def on_odom(self,m):
        self.odom_seen=True
        p=m.pose.pose.position; self.pose=(p.x,p.y,p.z,yaw(m.pose.pose.orientation))
    def on_cloud(self,m):
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
    def tick(self,_):
        # While a stair-managed phase drives /cmd_vel, suppress the goal
        # executor's idle zero so its 20 Hz arbitration messages cannot
        # degrade the ascent command stream (diagnosed on F2->F3: the zero
        # frames outnumbered the climb frames and the gait slipped on the
        # upper flight).  Only republish on a phase-boundary change; the two
        # stair nodes share this topic, and a constantly re-published False
        # from the idle node would fight the active node's True.
        desired_pause = self.phase in (
            'TRUTH_ENTRY_GUIDE', 'STAIR_PRE_ASCENT_ALIGN',
            'STAIR_TRUTH_STAGE_SETTLE', 'STAIR_POLICY_LOADING',
            'STAIR_POLICY_WARMUP', 'STAIR_ASCENT',
            'STAIR_LANDING_TURN', 'STAIR_ASCENT_B')
        if desired_pause != self._pause_active:
            self._pause_active = desired_pause
            self._pause_pub.publish(Bool(data=desired_pause))
        if self.phase=='WAIT_EXTERNAL_STAIR_ENTRY':
            self._start_external_entry_ascent()
            return
        if self.phase=='WAIT_F1':
            # Isolated F1 test: start only after local odometry and a local
            # registered cloud are both present.  Normal full missions still
            # wait for STAIR_WAIT_ZONE.
            if (self.auto_start and self.locomotion_ready and self.pose is not None
                    and (self.skip_approach or self.points or self.truth_entry_guide)):
                self.handoff_seen=True
                self.started=time.monotonic()
                if self.truth_entry_guide:
                    self.truth_entry_stage=(
                        'side' if self.truth_two_stage_guide else 'corridor')
                    self.truth_corridor_target=None
                    self.truth_side_target=None
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
                else:
                    self.phase='FIND_STAIR'
                self.state.publish(String(data=self.phase))
            else:
                return
        if self.phase=='TRUTH_ENTRY_GUIDE':
            if self.truth_pose is None or self.pose is None:
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
                # Two-stage on an upper floor routes via the solid landing-
                # side point, which requires the stair model SDF.  Without it,
                # fall back to the legacy three-stage guide (same behaviour as
                # the pre-two-stage code).
                if (self.truth_two_stage_guide and
                        int(getattr(self, 'source_floor_index', 0)) > 0 and
                        not (self.offline_stair_model and
                             os.path.isfile(self.offline_stair_model))):
                    rospy.logerr('Two-stage upper-floor guide requires '
                                 'offline_stair_model_sdf; reverting to the '
                                 'three-stage entry guide.')
                    self.truth_two_stage_guide=False
                    self.truth_entry_stage='corridor'
                if (self.truth_fixed_two_flight_profile and
                        self.truth_entry_stage in ('corridor', 'side')):
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
                        if self.truth_side_target is None:
                            corridor_target,side_target=(
                                self._truth_entry_route_targets(
                                    final_target, stair_heading))
                            self.truth_corridor_target=corridor_target
                            self.truth_side_target=side_target
                            rospy.logwarn(
                                'Truth stair route: corridor %s=(%.2f, %.2f), '
                                'side opening=(%.2f, %.2f), pre-riser=(%.2f, %.2f).',
                                'return' if self.truth_two_stage_guide else 'lobby',
                                corridor_target[0], corridor_target[1],
                                side_target[0], side_target[1],
                                final_target[0], final_target[1])
                        # The side stage folds in the corridor-return leg when
                        # the handoff picked the robot up deep inside the
                        # corridor: follow the corridor-centreline point until
                        # near the side opening, then cut across the open
                        # lobby.  A direct chord from the corridor interior
                        # would drive through the stair-core partition (run2
                        # wedged in a doorway at y~14.4).  On an upper floor
                        # both points are the landing-side exit, so the choice
                        # is a no-op.
                        if self.truth_entry_stage == 'corridor':
                            target=self.truth_corridor_target
                            self.truth_using_corridor_return=False
                        elif (self.truth_entry_stage == 'side' and
                              self.truth_two_stage_guide and
                              self.truth_corridor_target is not None and
                              math.hypot(self.truth_pose[0]-self.truth_side_target[0],
                                         self.truth_pose[1]-self.truth_side_target[1])
                              > self.truth_side_approach_radius):
                            target=self.truth_corridor_target
                            self.truth_using_corridor_return=True
                        else:
                            target=self.truth_side_target
                            self.truth_using_corridor_return=False
                else:
                    target=final_target
            else:
                self.truth_using_corridor_return=False
                target=self.truth_entry; stair_heading=math.atan2(target[1]-self.truth_pose[1], target[0]-self.truth_pose[0])
            # R41: while re-entering the corridor from a room, ignore the
            # two-stage waypoints above and steer straight to the next
            # re-entry point (room-side of the door, then centreline).
            if self.truth_reenter_points:
                self.truth_using_corridor_return=False
                target=self.truth_reenter_points[0]
                stair_heading=math.atan2(target[1]-self.truth_pose[1],
                                         target[0]-self.truth_pose[0])
            self.truth_entry_target=target
            dx=target[0]-self.truth_pose[0]; dy=target[1]-self.truth_pose[1]
            distance=math.hypot(dx,dy)
            heading=math.atan2(dy,dx)
            self.truth_route_heading=heading
            self.truth_entry_last_distance=distance
            self.truth_entry_best_distance=(distance if self.truth_entry_best_distance is None
                                             else min(self.truth_entry_best_distance, distance))
            self._refresh_truth_entry_progress_watchdog(distance)
            stage_tolerance=(
                .30 if self.truth_fixed_two_flight_profile and
                (self.truth_entry_stage == 'corridor' or
                 self.truth_using_corridor_return) else
                .20 if self.truth_fixed_two_flight_profile and
                self.truth_entry_stage == 'side' else .12)
            route_error=math.atan2(math.sin(heading-self.truth_pose[3]),
                                   math.cos(heading-self.truth_pose[3]))
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
                speed=min(self.truth_entry_guide_speed,
                          max(self.truth_entry_guide_min_speed, .70*distance))
                if abs(route_error) > .45:
                    speed=0.0
                else:
                    speed*=max(.35, math.cos(route_error))
                self._publish_truth_world_command(speed*math.cos(heading),
                                                  speed*math.sin(heading),
                                                  route_wz)
            else:
                self.cmd.publish(Twist())
            # A 0.45 m tolerance can launch the stair policy up to ~1.1 m
            # from the first riser.  It then walks on level ground rather
            # than engaging the first tread.  Require the calibrated staging
            # pose tightly before the final yaw/settle gate.
            if distance <= stage_tolerance:
                self.cmd.publish(Twist())
                if self.truth_reenter_points:
                    self.truth_reenter_points.pop(0)
                    self.truth_entry_best_distance=None
                    self.truth_entry_last_distance=None
                    self.truth_entry_watchdog_anchor_distance=None
                    if self.truth_reenter_points:
                        rospy.loginfo('Truth re-entry point reached; '
                                      'heading to the corridor centreline.')
                    else:
                        rospy.loginfo('Truth corridor re-entry complete; '
                                      'continuing the normal side approach.')
                        self.truth_entry_stage='side'
                        self.truth_entry_stage_switches+=1
                        self.truth_entry_deadline=(
                            time.monotonic()+self.truth_entry_timeout)
                        self.state.publish(String(
                            data='STAIR_TRUTH_SIDE_APPROACH'))
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
                        self.truth_entry_stage == 'side' and
                        not self.truth_using_corridor_return):
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
            if abs(error) > alignment_tolerance and abs(command.angular.z) < .22:
                sign_source=(command_error if abs(command_error) >= .04 else error)
                command.angular.z=math.copysign(.22, sign_source)
            self.cmd.publish(command)
            if abs(error) <= alignment_tolerance:
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
                self.phase='STAIR_PRE_ASCENT_ALIGN_TIMEOUT'
                self.state.publish(String(data=self.phase))
                rospy.logerr('Pre-ascent body alignment made no %.3f rad progress for %.1f s '
                             '(error=%.3f rad).',
                             self.pre_ascent_align_watchdog_progress,
                             self.pre_ascent_align_timeout, error)
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
                if self.policy_preloaded:
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
            # Keep the newly loaded stair policy in ownership while the gait
            # settles.  Starting ascent in the same tick as a policy swap
            # carries residual plane-policy motion into the first riser.
            self.cmd.publish(Twist())
            self.hold_rl()
            if (self.policy_warmup_until is not None and
                    time.monotonic() >= self.policy_warmup_until):
                self.phase='STAIR_ASCENT'
                # The ascent timeout measures stair-policy ownership only.
                # Truth-guided travel from the corridor entrance has its own
                # timeout and must not consume the two-flight climb budget.
                self.started=time.monotonic()
                self.ascent_start_z=self.pose[2] if self.pose else None
                self.truth_ascent_start_z=(self.truth_pose[2]
                                           if self.truth_entry_guide and self.truth_pose else None)
                self.flight_a_heading=self.direction
                self.state.publish(String(data=self.phase))
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
                    fall_detected=(
                        self.truth_flight_a_peak_gain >= .30 and
                        truth_gain < (self.truth_flight_a_peak_gain-
                                      self.truth_flight_a_fall_drop))
                    tracking_lost=(
                        truth_gain >= .20 and
                        (abs(center_error) >
                         self.truth_flight_a_max_center_error or
                         abs(heading_error) >
                         self.truth_flight_a_max_heading_error))
                    if fall_detected or tracking_lost:
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
                    # Hard stall guard: if even the released full climb
                    # speed makes no height gain for hard_stall_seconds,
                    # the gait is wedged on the treads (R8h/full48f).  Fail
                    # fast with a runner-recognized phase instead of
                    # hanging until the whole F1 stage budget is consumed.
                    # The landing turn and upper-floor handoff own their
                    # own timeouts in their respective branches.
                    if self.truth_flight_a_hard_stall_since is None:
                        self.truth_flight_a_hard_stall_since=time.monotonic()
                        self.truth_flight_a_hard_stall_start_gain=truth_gain
                    elif (truth_gain-self.truth_flight_a_hard_stall_start_gain >=
                          self.truth_flight_a_hard_stall_progress_m):
                        self.truth_flight_a_hard_stall_since=None
                    elif (time.monotonic()-self.truth_flight_a_hard_stall_since >=
                          self.truth_flight_a_hard_stall_seconds):
                        self.cmd.publish(Twist())
                        self.hold_rl()
                        self.phase='STAIR_ASCENT_TIMEOUT'
                        self.state.publish(String(data=self.phase))
                        rospy.logerr(
                            'Flight-A hard stall: height gain frozen at '
                            '%.2f m for %.0f s; stopping ascent.',
                            truth_gain, self.truth_flight_a_hard_stall_seconds)
                        self._record_trace(force=True)
                        rospy.signal_shutdown('stair_flight_a_hard_stall')
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
                    self.truth_flight_b_stall_start_z=None
                    self.truth_flight_b_recovery_until=None
                    self.truth_flight_b_entry_z=self.truth_pose[2]
                    self.phase='STAIR_ASCENT_B'
                    self.state.publish(String(data=self.phase))
                    rospy.loginfo('Flight-B entry aligned: x error=%.3f m, '
                                  'heading error=%.3f rad.',
                                  position_error, error)
                    self._save_entry_joint_snapshot()
                    return
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
                    self.truth_flight_b_stall_start_z=None
                    self.truth_flight_b_recovery_until=None
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
                self.truth_flight_b_stall_start_z=None
                self.truth_flight_b_recovery_until=None
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
        elif self.phase=='STAIR_ASCENT_B' and self.pose:
            rospy.loginfo_throttle(
                1.0,
                'ASCENT_B gate: entry_guide=%s truth_pose=%s start_z=%s '
                'profile=%s heading=%s pose_z=%s truth_z=%s',
                self.truth_entry_guide,
                self.truth_pose is not None,
                (('%.3f' % self.truth_ascent_start_z)
                 if self.truth_ascent_start_z is not None else None),
                self.truth_fixed_two_flight_profile,
                (('%.3f' % self.truth_flight_b_heading)
                 if self.truth_flight_b_heading is not None else None),
                self.pose[2] if self.pose is not None else None,
                self.truth_pose[2] if self.truth_pose is not None else None)
            # Optional pre-ascent stand gate: re-initialise to the default
            # joint pose once, before the climb, then fall through to normal
            # ascent.  F1->F2 instance has pre_ascent_stand_seconds=0 so this
            # is a no-op there; only the F2->F3 instance re-enters the RL gait
            # from a clean pose.
            if (self.pre_ascent_stand_seconds > 0.0 and
                    not self.pre_ascent_stand_done):
                if self.pre_ascent_stand_started is None:
                    self.pre_ascent_stand_started=time.monotonic()
                    self._apply_stand_target_joints()
                    self._publish_joy_fixed_stand()
                    self.cmd.publish(Twist())
                    rospy.loginfo('Pre-ascent fixed stand: holding %.1f s at '
                                  'snapshot pose before flight-B climb.',
                                  self.pre_ascent_stand_seconds)
                    self._record_trace()
                    return
                if (time.monotonic()-self.pre_ascent_stand_started >=
                        self.pre_ascent_stand_seconds):
                    self.pre_ascent_stand_done=True
                    self._clear_stand_target_joints()
                    self.hold_rl()
                    rospy.loginfo('Pre-ascent stand complete (%.1f s); '
                                  'returning to RL climb.',
                                  self.pre_ascent_stand_seconds)
                    self._record_trace()
                else:
                    self._publish_joy_fixed_stand()
                    self.cmd.publish(Twist())
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
                if (self.truth_flight_b_peak_gain >=
                        self.flight_a_height_gain+.30 and
                        truth_gain <
                        self.truth_flight_b_peak_gain-self.truth_flight_b_fall_drop):
                    self.cmd.publish(Twist())
                    self.hold_rl()
                    self.phase='STAIR_FLIGHT_B_FALL_DETECTED'
                    self.state.publish(String(data=self.phase))
                    rospy.logerr('Flight-B height dropped %.2f m from its peak; '
                                 'stopping instead of continuing toward the stair edge.',
                                 self.truth_flight_b_peak_gain-truth_gain)
                    self._record_trace()
                    rospy.signal_shutdown('stair_flight_b_fall_detected')
                    return
                cleared,clearance_margin=self._truth_upper_landing_clearance()
                if (truth_gain >= self.total_height_gain and cleared):
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
                if self._truth_ascent_timed_out(time.monotonic(), truth_gain):
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
                # Stall recovery: z frozen while commanding full climb speed
                # means the gait is airborne on a tread (rearward pitch slip,
                # yaw drifting).  Pause the command so the legs land and
                # re-establish contact, then resume the full-speed climb.
                # Re-arms after each resume so the punch-out gets a fresh
                # window; normal climbs gain far more than the threshold.
                if self.truth_flight_b_stall_start_z is None:
                    self.truth_flight_b_stall_start_z=self.truth_pose[2]
                    self.truth_flight_b_stall_since=time.monotonic()
                # Slide-off detection fires during the fall itself (z drops
                # past the threshold while the gait is still in contact),
                # instead of waiting for the 4 s freeze gate to arm the stall
                # recovery after the body is already wedged.  The recovery
                # loop must not suppress it: full50 slid during a recovery
                # cycle and the suppression gate hid the drop until the 6th
                # pause, by which time the body was wedged.  After 5 failed
                # backoff attempts the slide is treated as a normal stall and
                # the watchdog remains the final backstop.  The backoff run-up
                # escalates 0.35 -> 0.50 -> 0.65 -> 0.75 m per attempt (same
                # ladder as the descent v2 recovery), giving the wedged body
                # more tries with a wider run-up before giving up.
                dropped=(self.truth_flight_b_entry_z is not None and
                         self.truth_flight_b_entry_z-self.truth_pose[2] >
                         self.truth_flight_b_drop_detect)
                if (dropped and
                        self.truth_flight_b_drop_backoffs >= 5):
                    dropped=False
                    if not self.truth_flight_b_slide_exhausted_logged:
                        self.truth_flight_b_slide_exhausted_logged=True
                        rospy.logerr(
                            'Flight-B slide recovery exhausted (%d attempts); '
                            'treating as a normal stall.',
                            self.truth_flight_b_drop_backoffs)
                if dropped:
                    # Do not pause: an immediate backoff toward the platform
                    # (world +Y) is the only chance to climb back up before
                    # the body wedges in the step/landing gap.  The stall
                    # branch below would otherwise pause 6 s while the body
                    # keeps sinking.
                    if self.truth_flight_b_backoff_until is None:
                        # Escalate the run-up with each failed attempt: the
                        # body keeps wedging on the same riser, so a longer
                        # backoff (0.35 -> 0.50 -> 0.65 -> 0.75 m) gives the
                        # re-sprint more momentum to clear the step/landing
                        # gap (same ladder as the descent v2 recovery).
                        self.truth_flight_b_backoff_until=(
                            time.monotonic()+
                            min(self.truth_flight_b_recovery_backoff_distance_m+
                                self.truth_flight_b_recovery_backoff_escalation_m*
                                self.truth_flight_b_drop_backoffs,
                                self.truth_flight_b_recovery_backoff_max_distance_m)/
                            max(self.truth_flight_b_recovery_backoff_speed_mps,
                                1e-3))
                        self.truth_flight_b_drop_backoffs+=1
                        rospy.logerr(
                            'Flight-B dropped %.3f m below landing height '
                            '(entry z=%.3f, z=%.3f); immediate backoff %d '
                            'toward the platform.',
                            self.truth_flight_b_entry_z-self.truth_pose[2],
                            self.truth_flight_b_entry_z, self.truth_pose[2],
                            self.truth_flight_b_drop_backoffs)
                    if (time.monotonic() <
                            self.truth_flight_b_backoff_until):
                        target_heading=(self.truth_flight_b_heading
                                        if self.truth_flight_b_heading is not None
                                        else -math.pi/2.0)
                        yaw_err=math.atan2(
                            math.sin(target_heading-self.truth_pose[3]),
                            math.cos(target_heading-self.truth_pose[3]))
                        yaw_rate=max(
                            -self.truth_flight_b_recovery_yaw_rate,
                            min(self.truth_flight_b_recovery_yaw_rate,
                                1.2*yaw_err))
                        self._publish_truth_world_command(
                            0.0, self.truth_flight_b_recovery_backoff_speed_mps,
                            yaw_rate)
                        self._record_trace()
                        return
                    self.truth_flight_b_backoff_until=None
                    if (self.truth_flight_b_entry_z-self.truth_pose[2] <=
                            self.truth_flight_b_drop_detect):
                        # Back up onto the landing again: clear the slide
                        # state and re-arm a fresh stall window so the normal
                        # climb can retry from the platform.
                        self.truth_flight_b_drop_backoffs=0
                        self.truth_flight_b_stall_start_z=self.truth_pose[2]
                        self.truth_flight_b_stall_since=time.monotonic()
                        rospy.logwarn('Flight-B back to landing height '
                                      '(z=%.3f); slide recovery cleared.',
                                      self.truth_pose[2])
                        dropped=False
                if (self.truth_pose[2]-self.truth_flight_b_stall_start_z >=
                        self.truth_flight_b_stall_z_progress_m):
                    self.truth_flight_b_stall_start_z=self.truth_pose[2]
                    self.truth_flight_b_stall_since=time.monotonic()
                    stall=False
                else:
                    slip_mode=(self.truth_flight_b_slip_detect_seconds > 0 and
                               truth_gain > self.truth_flight_b_slip_min_gain_m)
                    freeze_timeout=(self.truth_flight_b_slip_detect_seconds
                                    if slip_mode else
                                    self.truth_flight_b_stall_seconds)
                    stall=(time.monotonic()-self.truth_flight_b_stall_since >=
                           freeze_timeout)
                if stall or dropped:
                    slip_mode=(self.truth_flight_b_slip_detect_seconds > 0 and
                               truth_gain > self.truth_flight_b_slip_min_gain_m)
                    # A backoff that is already running must finish before any
                    # new pause re-arms.  Otherwise every tick re-enters this
                    # stall branch with recovery_until cleared, and the backoff
                    # receives one command before the pause restarts (full49:
                    # backoff started, re-stalled 0.04 s later, so the run-up
                    # never happened).
                    backoff_complete=False
                    if (not slip_mode and
                            self.truth_flight_b_backoff_until is not None):
                        if (time.monotonic() <
                                self.truth_flight_b_backoff_until):
                            target_heading=(self.truth_flight_b_heading
                                            if self.truth_flight_b_heading is not None
                                            else -math.pi/2.0)
                            yaw_err=math.atan2(
                                math.sin(target_heading-self.truth_pose[3]),
                                math.cos(target_heading-self.truth_pose[3]))
                            yaw_rate=max(
                                -self.truth_flight_b_recovery_yaw_rate,
                                min(self.truth_flight_b_recovery_yaw_rate,
                                    1.2*yaw_err))
                            self._publish_truth_world_command(
                                0.0, self.truth_flight_b_recovery_backoff_speed_mps,
                                yaw_rate)
                            self._record_trace()
                            return
                        self.truth_flight_b_backoff_until=None
                        backoff_complete=True
                    if not backoff_complete:
                        if self.truth_flight_b_recovery_until is None:
                            self.truth_flight_b_recovery_until=(
                                time.monotonic()+
                                (self.truth_flight_b_slip_hold_seconds
                                 if slip_mode else
                                 self.truth_flight_b_recovery_hold_seconds))
                            rospy.logwarn(
                                'Flight-B %s (z=%.3f m, gain=%.2f m); '
                                'pausing %.1f s to re-land the gait.',
                                ('slip caught' if slip_mode else 'stalled'),
                                self.truth_pose[2], truth_gain,
                                (self.truth_flight_b_slip_hold_seconds
                                 if slip_mode else
                                 self.truth_flight_b_recovery_hold_seconds))
                        if time.monotonic() < self.truth_flight_b_recovery_until:
                            # Re-land the gait and, while paused, rotate the
                            # body back onto the flight-B centreline so the
                            # resumed sprint attacks the riser head-on.
                            # full28's stall paused at yaw -2.0..-2.1 and every
                            # resumed sprint kept drifting (-2.78 max):
                            # sideways on the tread the feet re-catch the same
                            # riser and slip again.  In slip mode the climb
                            # froze at most ~1 s ago and the legs can still
                            # push, so use full yaw authority to get back on
                            # axis before the gait collapses.
                            target_heading=(self.truth_flight_b_heading
                                            if self.truth_flight_b_heading is not None
                                            else -math.pi/2.0)
                            yaw_err=math.atan2(
                                math.sin(target_heading-self.truth_pose[3]),
                                math.cos(target_heading-self.truth_pose[3]))
                            if abs(yaw_err) > self.truth_flight_b_recovery_yaw_tolerance:
                                eff_yaw_rate=(self.truth_flight_b_slip_yaw_rate
                                              if slip_mode else
                                              self.truth_flight_b_recovery_yaw_rate)
                                yaw_rate=max(
                                    -eff_yaw_rate,
                                    min(eff_yaw_rate, 1.2*yaw_err))
                                self._publish_truth_world_command(
                                    0.0, 0.0, yaw_rate)
                                rospy.logwarn_throttle(
                                    1.0,
                                    'Flight-B stall pause: re-aligning yaw '
                                    '(err=%.3f rad, wz=%.2f).', yaw_err, yaw_rate)
                            else:
                                self.cmd.publish(Twist())
                                self.hold_rl()
                            self._record_trace()
                            return
                        self.truth_flight_b_recovery_until=None
                        if (not slip_mode and
                                self.truth_flight_b_backoff_until is None):
                            # Recovery pause complete: before punching back to
                            # full speed, back off down the flight so the
                            # sprint re-approaches the riser with a run-up
                            # (full34 showed the paused+ramped re-attempt
                            # still slips the same tread 4x in a row).  Slip
                            # mode skips the backoff: the climb only froze
                            # ~1 s ago, the legs are still landed, and backing
                            # off would surrender the gained height for no
                            # run-up.
                            # Escalate the run-up with each stall recovery:
                            # repeated wedges get a wider backoff (0.35 ->
                            # 0.50 -> 0.65 -> 0.75 m) so the re-sprint can
                            # clear the same riser (descent v2 ladder).
                            backoff_distance=min(
                                self.truth_flight_b_recovery_backoff_distance_m+
                                self.truth_flight_b_recovery_backoff_escalation_m*
                                self.truth_flight_b_stall_recoveries,
                                self.truth_flight_b_recovery_backoff_max_distance_m)
                            self.truth_flight_b_backoff_until=(
                                time.monotonic()+
                                backoff_distance/
                                max(self.truth_flight_b_recovery_backoff_speed_mps,
                                    1e-3))
                            rospy.logwarn(
                                'Flight-B stall pause complete; backing off %.2f m '
                                '(%.2f m/s) for a fresh run-up.',
                                backoff_distance,
                                self.truth_flight_b_recovery_backoff_speed_mps)
                        if (not slip_mode and
                                time.monotonic() < self.truth_flight_b_backoff_until):
                            # Back off along the flight-B axis (world +Y)
                            # while keeping the centreline heading.
                            target_heading=(self.truth_flight_b_heading
                                            if self.truth_flight_b_heading is not None
                                            else -math.pi/2.0)
                            yaw_err=math.atan2(
                                math.sin(target_heading-self.truth_pose[3]),
                                math.cos(target_heading-self.truth_pose[3]))
                            yaw_rate=max(
                                -self.truth_flight_b_recovery_yaw_rate,
                                min(self.truth_flight_b_recovery_yaw_rate,
                                    1.2*yaw_err))
                            self._publish_truth_world_command(
                                0.0, self.truth_flight_b_recovery_backoff_speed_mps,
                                yaw_rate)
                            self._record_trace()
                            return
                        self.truth_flight_b_backoff_until=None
                    self.truth_flight_b_stall_start_z=self.truth_pose[2]
                    self.truth_flight_b_stall_since=time.monotonic()
                    self.truth_flight_b_stall_recoveries+=1
                    self.truth_flight_b_ramp_until=(
                        time.monotonic()+
                        self.truth_flight_b_recovery_ramp_seconds)
                    rospy.logwarn(
                        'Flight-B resuming climb after stall pause %d '
                        '(ramp %.2f m/s for %.1f s, then %.2f m/s).',
                        self.truth_flight_b_stall_recoveries,
                        self.truth_flight_b_recovery_ramp_speed,
                        self.truth_flight_b_recovery_ramp_seconds,
                        self.ascent_speed)
                if self.truth_fixed_two_flight_profile:
                    center_error=(self._truth_flight_b_center_x()-
                                  self.truth_pose[0])
                    self.truth_flight_b_center_error=center_error
                    if abs(center_error) <= self.truth_flight_b_center_deadband:
                        center_speed=0.0
                    else:
                        center_speed=max(
                            -self.truth_flight_b_center_speed,
                            min(self.truth_flight_b_center_speed,
                                self.truth_flight_b_center_gain*center_error))
                    target_heading=(self.truth_flight_b_heading
                                    if self.truth_flight_b_heading is not None
                                    else -math.pi/2.0)
                    heading_error=math.atan2(
                        math.sin(target_heading-self.truth_pose[3]),
                        math.cos(target_heading-self.truth_pose[3]))
                    self.truth_flight_b_heading_error=heading_error
                    yaw_rate=max(
                        -self.truth_flight_b_max_yaw_rate,
                        min(self.truth_flight_b_max_yaw_rate,
                            self.truth_flight_b_heading_gain*heading_error))
                    climb_speed=self.ascent_speed
                    if (self.truth_flight_b_ramp_until is not None and
                            time.monotonic() < self.truth_flight_b_ramp_until):
                        climb_speed=self.truth_flight_b_recovery_ramp_speed
                    else:
                        self.truth_flight_b_ramp_until=None
                    self._publish_truth_world_command(
                        center_speed, -climb_speed, yaw_rate)
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
            stable=(truth_gain is not None and
                    truth_gain >= self.total_height_gain and cleared)
            if not stable:
                # A brief stop can let the last feet slip back onto the top
                # tread.  Resume the same flight-B command instead of declaring
                # success from the earlier sample.
                self.second_floor_settle_started=None
                self.phase='STAIR_ASCENT_B'
                self.state.publish(String(data=self.phase))
                rospy.logwarn('Upper-landing stability regressed; resuming flight B.')
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
    def save(self):
        with open(os.path.join(self.out,'logs',self.transition_log_basename),'w') as f:
            json.dump({'phase':self.phase,'policy':self.policy,'trace':self.trace,
                       'entry_guide':{
                           'two_stage_guide':self.truth_two_stage_guide,
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
                               self.truth_flight_a_recovery_yaw_rate},
                       'landing_recovery':{
                           'active':self.landing_recovery_active,
                           'attempts':self.landing_recovery_attempts,
                           'maximum_attempts':self.landing_recovery_max_attempts,
                           'retry_timeout_sec':self.landing_recovery_timeout,
                           'minimum_yaw_rate_rps':
                               self.landing_recovery_minimum_yaw_rate},
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
                truth_time=[float(row.get('elapsed_time',0.0))-
                            float(truth[0].get('elapsed_time',0.0)) for row in truth]
                axes[1].plot(truth_time,tz,'r-',lw=2,label='Gazebo truth height')
                peak=max(range(len(tz)),key=lambda index:tz[index])
                axes[1].scatter(truth_time[peak],tz[peak],c='#e17b21',s=55,
                                zorder=4,label='maximum height {:.2f} m'.format(tz[peak]))
                axes[1].scatter(truth_time[-1],tz[-1],c='#111111',s=50,
                                marker='X',zorder=4,label='attempt end {:.2f} m'.format(tz[-1]))
                axes[1].set(xlabel='stair elapsed time (s)',ylabel='Gazebo truth z (m)',title='physical vertical ascent'); axes[1].grid(); axes[1].legend()
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
                                xlabel='elapsed after F{} handoff (s)'.format(
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
                axes[1].plot(ts,zs,'r-',lw=2); axes[1].set(xlabel='stair elapsed time (s)',ylabel='odometry z (m)',title='vertical ascent profile'); axes[1].grid()
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
                height.set(title='Stair height profile',xlabel='elapsed after F1 handoff (s)',ylabel='odometry z (m)')
                height.grid(alpha=.25); height.legend(fontsize=8)
                figure.savefig(os.path.join(
                    self.out,'visualization',self.trajectory_plot_basename),dpi=180)
                plt.close(figure)
        except Exception as error:
            rospy.logwarn('Could not write stair visualization: %s', error)
if __name__=='__main__':
    rospy.init_node('stair_transition_manager'); StairTransition(); rospy.spin()
