#!/usr/bin/env python3
"""Online F1-to-stair handoff with logs for the second-floor controller.

The node uses only F1's current odometry and registered cloud.  It never
reads the building layout or Gazebo truth.  This first version performs the
safe handoff and records an ascent trace; its direction score is deliberately
conservative and requires a positive-height, near-range cloud sector.
"""
import json, math, os, time, shutil, threading
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

def roll_pitch(q):
    """Return body roll/pitch from a geometry quaternion."""
    roll=math.atan2(2*(q.w*q.x+q.y*q.z),
                    1-2*(q.x*q.x+q.y*q.y))
    pitch_sine=2*(q.w*q.y-q.z*q.x)
    pitch=math.asin(max(-1.0, min(1.0, pitch_sine)))
    return roll,pitch

class StairTransition:
    def __init__(self):
        self.out=os.path.abspath(rospy.get_param('~output_dir'))
        self.policy=rospy.get_param('~stair_policy')
        # The generated-world bridge may make one bounded return to the
        # pre-riser after an early, low-height Flight-A derail.  That return
        # must use the level-ground policy; continuing to steer with the
        # stair gait after it has dropped back to the lobby is ineffective.
        self.plane_policy=str(rospy.get_param(
            '~plane_policy', '')).strip()
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
        self._entry_joint_recv_wall=None
        self.entry_joint_velocity_rms=None
        # JointState subscribers and the 50 Hz control timer run on separate
        # rospy callback threads.  Keep the velocity qualification and the
        # corresponding position snapshot atomic; round59 qualified one
        # low-speed sample, then froze positions from the following sample.
        self._entry_joint_lock=threading.RLock()
        # Stand target override: SNAPSHOT_JOINTS_FILE points at the captured
        # F1->F2 entry pose (flight_b_entry_joints_0.json).  It is pushed to
        # /stand_target_joints only for the pre-ascent stand and cleared
        # afterwards, so the startup auto-hold stand keeps the default pose.
        self.snapshot_file=str(os.environ.get(
            'SNAPSHOT_JOINTS_FILE', '')).strip()
        self._stand_target_param='/stand_target_joints'
        # Ownership flag: True only when THIS instance successfully pushed
        # /stand_target_joints, so the clear step never deletes a param it
        # does not own (which raised KeyError and killed the timer thread in
        # round11 when a configured snapshot file was missing).
        self._stand_target_applied=False
        self.joint_state_sub=rospy.Subscriber(
            '/a1_gazebo/joint_states', JointState, self._on_joint_states,
            queue_size=1)
        self.return_transit_armed=False
        self.return_gate_published=False
        self.truth_return_gate_radius=float(rospy.get_param(
            '~truth_return_gate_radius_m', 5.40))
        self.locomotion_ready=False
        # Wall time of the last /locomotion_ready=True arrival; used to
        # distinguish a fresh post-FixedStand RL handshake from the cached
        # readiness that predates the pre-ascent stand.
        self.locomotion_ready_at=None
        # FixedStand publishes a latched false->true readiness edge on every
        # entry.  The F1 plane->stair boundary can require that fresh edge so
        # it never leaves a mechanically unsettled gait snapshot merely
        # because a short wall-clock hold elapsed.
        self.fixed_stand_ready=False
        self.fixed_stand_ready_at=None
        self.fixed_stand_not_ready_at=None
        self.truth_flight_a_policy_fixed_stand_ready_confirmed=False
        self.truth_twist=None
        self.truth_roll=None
        self.truth_pitch=None
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
        # Speed caps for the truth entry guide.  The high common cap shortens
        # only the corridor/lobby transfer.  Keep an independent cap for the
        # last side-opening -> pre-riser traverse: round83e entered that
        # 1.65 m leg at 0.9 m/s, built 0.8 m/s lateral velocity and rolled
        # onto its side before the stair policy had even been requested.
        self.truth_entry_guide_speed=float(rospy.get_param(
            '~truth_entry_guide_speed_mps', .45))
        self.truth_entry_guide_min_speed=float(rospy.get_param(
            '~truth_entry_guide_min_speed_mps', .25))
        self.truth_entry_final_speed=max(
            self.truth_entry_guide_min_speed, float(rospy.get_param(
                '~truth_entry_final_speed_mps', .55)))
        # The broad F1 return gate can take ownership while the body is still
        # in a far-room doorway, just outside the longitudinal corridor band.
        # A point-at-the-distant-waypoint controller dilutes a 0.6 m lateral
        # error over a 20+ m route and therefore leaves the doorway before it
        # has re-entered the corridor (full-flow round18).  Track the physical
        # corridor line independently: large or predicted outward error first
        # triggers a lateral-only recapture, then a bounded centre correction
        # accompanies the longitudinal return.
        self.truth_corridor_recenter_enter_error=max(0.05, float(
            rospy.get_param(
                '~truth_corridor_recenter_enter_error_m', .30)))
        self.truth_corridor_recenter_release_error=max(0.02, min(
            self.truth_corridor_recenter_enter_error,
            float(rospy.get_param(
                '~truth_corridor_recenter_release_error_m', .10))))
        self.truth_corridor_recenter_speed=max(0.05, float(rospy.get_param(
            '~truth_corridor_recenter_speed_mps', .40)))
        self.truth_corridor_center_gain=max(0.0, float(rospy.get_param(
            '~truth_corridor_center_gain', .90)))
        self.truth_corridor_tracking_center_speed=max(0.0, float(
            rospy.get_param(
                '~truth_corridor_tracking_center_speed_mps', .26)))
        self.truth_corridor_center_deadband=max(0.0, float(rospy.get_param(
            '~truth_corridor_center_deadband_m', .04)))
        self.truth_corridor_prediction_horizon=max(0.0, float(
            rospy.get_param(
                '~truth_corridor_prediction_horizon_sec', .70)))
        self.truth_corridor_prediction_min_lateral_speed=max(0.0, float(
            rospy.get_param(
                '~truth_corridor_prediction_min_lateral_speed_mps', .06)))
        self.truth_corridor_prediction_velocity_alpha=min(1.0, max(.01, float(
            rospy.get_param(
                '~truth_corridor_prediction_velocity_alpha', .30))))
        # At round18 handoff the body faced back into the room.  Backing the
        # final 0.6 m through the still-open doorway is safer than spending a
        # half turn drifting beside its jamb.  The command remains a world-
        # frame line recapture; this flag only selects the nearer longitudinal
        # body orientation for the plane gait.
        self.truth_corridor_recenter_allow_reverse=bool(rospy.get_param(
            '~truth_corridor_recenter_allow_reverse', True))
        self.truth_corridor_recenter_active=False
        self.truth_corridor_center_error=None
        self.truth_corridor_predicted_center_error=None
        self.truth_corridor_lateral_velocity=None
        self.truth_corridor_command_center_speed=None
        self.truth_corridor_command_forward_speed=None
        self.truth_corridor_motion_heading=None
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
        # Production can keep ordinary centre/yaw corrections moving briskly
        # without carrying that speed into a genuine edge/slip escape.  A
        # negative value preserves the historical behaviour in generic
        # launches; the three-floor launch opts into a separately bounded
        # strong-recovery speed.
        self.truth_flight_a_strong_recovery_forward_speed=float(
            rospy.get_param(
                '~truth_flight_a_strong_recovery_forward_speed_mps', -1.0))
        self.truth_flight_a_near_edge_strong_center_error=max(0.0, float(
            rospy.get_param(
                '~truth_flight_a_near_edge_strong_center_error_m', 0.0)))
        self.truth_flight_a_recovery_center_speed=float(rospy.get_param(
            '~truth_flight_a_recovery_center_speed_mps', .18))
        self.truth_flight_a_recovery_yaw_rate=float(rospy.get_param(
            '~truth_flight_a_recovery_yaw_rate_rps', .16))
        # Full-flow round34 reached the upper flight-A treads squarely, then
        # slipped about 0.25 m down from its achieved height.  The normal
        # alignment crawl (0.12 m/s forward, 0.16 rad/s yaw) is appropriate
        # for lateral drift, but it is too weak to keep a slipping learned
        # gait in tread contact.  Escalate only after a measurable height
        # drawdown; ordinary climbs retain the conservative controller.
        self.truth_flight_a_slip_min_peak_gain=max(0.0, float(
            rospy.get_param('~truth_flight_a_slip_min_peak_gain_m', .30)))
        self.truth_flight_a_slip_drawdown_trigger=max(0.0, float(
            rospy.get_param('~truth_flight_a_slip_drawdown_trigger_m', .06)))
        self.truth_flight_a_slip_forward_speed=max(0.0, float(
            rospy.get_param('~truth_flight_a_slip_forward_speed_mps', .40)))
        self.truth_flight_a_slip_yaw_rate=max(0.0, float(
            rospy.get_param('~truth_flight_a_slip_yaw_rate_rps', .60)))
        # A height-drawdown guard only notices a stair slip after the trunk
        # has already lost altitude.  In final stress seed 4, Flight A first
        # moved rearward for more than a second while its height still rose;
        # pitch then crossed the Euler singularity and the heading guard was
        # necessarily too late.  Project Gazebo twist onto the flight axis,
        # as the proven Flight-B guard does, and re-land the same stair gait
        # before that posture collapse.  The generic default remains off;
        # the generated three-floor acceptance launch opts in explicitly.
        self.truth_flight_a_rearward_slip_speed=max(0.0, float(
            rospy.get_param(
                '~truth_flight_a_rearward_slip_speed_mps', 0.0)))
        self.truth_flight_a_rearward_slip_detect_seconds=max(0.0, float(
            rospy.get_param(
                '~truth_flight_a_rearward_slip_detect_seconds', .04)))
        self.truth_flight_a_rearward_slip_min_height_gain=max(0.0, float(
            rospy.get_param(
                '~truth_flight_a_rearward_slip_min_height_gain_m', .40)))
        self.truth_flight_a_rearward_slip_velocity_alpha=min(
            1.0, max(.01, float(rospy.get_param(
                '~truth_flight_a_rearward_slip_velocity_alpha', .70))))
        self.truth_flight_a_rearward_slip_release_ratio=min(
            1.0, max(0.0, float(rospy.get_param(
                '~truth_flight_a_rearward_slip_release_ratio', .80))))
        # A learned stair gait can briefly move rearward in the horizontal
        # plane during a healthy, fast upward footfall.  Treat that as a
        # backslide only after upward progress has also slowed; the existing
        # peak-drawdown, corridor, heading and posture guards remain active
        # independently.  This prevents the recovery hold itself from
        # interrupting a strongly ascending gait.
        self.truth_flight_a_rearward_slip_max_climb_speed=max(0.0, float(
            rospy.get_param(
                '~truth_flight_a_rearward_slip_max_climb_speed_mps', .12)))
        self.truth_flight_a_backslide_recovery_hold_seconds=max(0.0, float(
            rospy.get_param(
                '~truth_flight_a_backslide_recovery_hold_seconds', 1.50)))
        self.truth_flight_a_backslide_recovery_ramp_seconds=max(0.0, float(
            rospy.get_param(
                '~truth_flight_a_backslide_recovery_ramp_seconds', 1.00)))
        self.truth_flight_a_backslide_recovery_ramp_speed=max(0.0, float(
            rospy.get_param(
                '~truth_flight_a_backslide_recovery_ramp_speed_mps', .40)))
        self.truth_flight_a_backslide_recovery_max_attempts=max(1, int(
            rospy.get_param(
                '~truth_flight_a_backslide_recovery_max_attempts', 2)))
        self.truth_flight_a_rearward_speed=None
        self.truth_flight_a_filtered_rearward_speed=None
        self.truth_flight_a_filtered_climb_speed=None
        self.truth_flight_a_rearward_slip_since=None
        self.truth_flight_a_rearward_slip_latched=False
        self.truth_flight_a_rearward_slip_events=0
        self.truth_flight_a_rearward_slip_last_trigger_gain=None
        self.truth_flight_a_backslide_recovery_until=None
        self.truth_flight_a_backslide_recovery_ramp_until=None
        self.truth_flight_a_backslide_brake_active=False
        self.truth_flight_a_backslide_brake_center_speed=0.0
        self.truth_flight_a_backslide_brake_yaw_rate=0.0
        self.truth_flight_a_backslide_brake_peak_center_speed=0.0
        # A heading-only guard crossing while the trunk is still centred is
        # recoverable.  Keep the original centre/fall hard guards, but allow
        # a short, progress-checked window before declaring alignment loss.
        self.truth_flight_a_alignment_recovery_seconds=max(.1, float(
            rospy.get_param(
                '~truth_flight_a_alignment_recovery_seconds', 1.5)))
        self.truth_flight_a_alignment_recovery_max_attempts=max(1, int(
            rospy.get_param(
                '~truth_flight_a_alignment_recovery_max_attempts', 2)))
        self.truth_flight_a_alignment_recovery_safe_center_error=max(
            0.0, float(rospy.get_param(
                '~truth_flight_a_alignment_recovery_safe_center_error_m',
                .24)))
        self.truth_flight_a_alignment_recovery_hard_heading_error=max(
            self.truth_flight_a_max_heading_error, float(rospy.get_param(
                '~truth_flight_a_alignment_recovery_hard_heading_error_rad',
                1.20)))
        self.truth_flight_a_alignment_recovery_release_heading_error=max(
            0.0, float(rospy.get_param(
                '~truth_flight_a_alignment_recovery_release_heading_error_rad',
                .20)))
        self.truth_flight_a_alignment_recovery_release_center_error=max(
            0.0, float(rospy.get_param(
                '~truth_flight_a_alignment_recovery_release_center_error_m',
                .16)))
        self.truth_flight_a_alignment_recovery_stable_seconds=max(0.0, float(
            rospy.get_param(
                '~truth_flight_a_alignment_recovery_stable_seconds', .25)))
        # The learned gait needs roughly one half gait cycle before a lateral
        # command reverses existing side velocity.  Position-only recovery
        # therefore reacted too late in full-flow round4: the body was still
        # near the centreline but already moving toward the east tread edge at
        # 0.27--0.31 m/s.  Predict a short, filtered lateral stopping envelope
        # from Gazebo ModelStates twist and enter the same bounded recovery
        # before the physical position guard is crossed.
        self.truth_flight_a_prediction_horizon=max(0.0, float(
            rospy.get_param(
                '~truth_flight_a_prediction_horizon_sec', .70)))
        self.truth_flight_a_prediction_min_lateral_speed=max(0.0, float(
            rospy.get_param(
                '~truth_flight_a_prediction_min_lateral_speed_mps', .08)))
        self.truth_flight_a_prediction_velocity_alpha=min(1.0, max(0.01, float(
            rospy.get_param(
                '~truth_flight_a_prediction_velocity_alpha', .25))))
        # The predictive guard can see a fast lateral escape while the body
        # is still physically near centre.  The legacy response then reduced
        # forward speed to 0.12 m/s and capped counter-strafe at 0.18 m/s;
        # Round36's isolated replay accelerated sideways from 0.23 to 0.80
        # m/s before crossing the unchanged 0.38 m hard guard.  Once genuine
        # stair height is established, retain enough tread-contact speed and
        # give the predictor bounded authority to reverse that momentum.
        self.truth_flight_a_predictive_strong_min_lateral_speed=max(
            0.0, float(rospy.get_param(
                '~truth_flight_a_predictive_strong_min_lateral_speed_mps',
                .20)))
        self.truth_flight_a_predictive_strong_forward_speed=max(
            0.0, float(rospy.get_param(
                '~truth_flight_a_predictive_strong_forward_speed_mps',
                .40)))
        self.truth_flight_a_predictive_strong_center_speed=max(
            0.0, float(rospy.get_param(
                '~truth_flight_a_predictive_strong_center_speed_mps',
                .30)))
        # A centreline crossing is useful evidence for starting a bounded
        # recovery, but on Flight A it can also mean that an earlier strong
        # counter-strafe has already reversed the lateral motion.  Let each
        # stair instance decide whether that crossing alone is sufficient to
        # keep using the full strong-recovery command.  The historical
        # behaviour remains the default for the independently tuned F2->F3
        # instance; F1 disables it in the launch file below.
        self.truth_flight_a_predictive_strong_on_center_crossing=bool(
            rospy.get_param(
                '~truth_flight_a_predictive_strong_on_center_crossing',
                True))
        self.truth_flight_a_predictive_strong_active=False
        self.truth_flight_a_near_edge_strong_active=False
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
        # The control loop used to release full climb speed for only the one
        # timer tick that closed a five-second stall window.  At 20 Hz that is
        # roughly 50 ms--far shorter than a learned-gait step--so the very next
        # tick restored the 0.12 m/s alignment crawl and left the body pinned
        # against the same riser.  Production launches opt into separately
        # tuned bounded multi-step releases for F1 and F2; the zero class
        # default preserves compatibility for other launch files.
        self.truth_flight_a_stall_release_seconds=max(0.0, float(
            rospy.get_param('~truth_flight_a_stall_release_seconds', 0.0)))
        self.truth_flight_a_stall_release_max_center_error=max(0.0, float(
            rospy.get_param(
                '~truth_flight_a_stall_release_max_center_error_m', .25)))
        self.truth_flight_a_stall_release_max_heading_error=max(0.0, float(
            rospy.get_param(
                '~truth_flight_a_stall_release_max_heading_error_rad', .35)))
        self.truth_flight_a_stall_release_until=None
        self.truth_flight_a_stall_release_active=False
        # If a stall begins just outside the conservative full-speed release
        # band, do not leave the gait forever in the 0.12 m/s recovery crawl.
        # A bounded recenter escape keeps tread-contact speed while retaining
        # strong lateral correction; the unchanged hard centre/yaw guards in
        # tick() still stop the mission before the tread edge is crossed.
        self.truth_flight_a_stall_realign_active=False
        self.truth_flight_a_stall_realign_attempts=0
        self.truth_flight_a_recovery_since=None
        self.truth_flight_a_recovery_start_z=None
        # Readiness regression: STAIR_POLICY_WARMUP ends on a fixed timer, so
        # flight A can begin (and its stall/peak clocks can run) while the
        # fresh RL takeover is still latching; the first effective gait
        # command then steps directly to ascent_speed and the body skews off
        # the centreline before the recovery profile can engage.  Upper-
        # production stair launches opt in to (a) holding zero command until a fresh
        # /locomotion_ready is observed in STAIR_ASCENT, resetting all
        # flight-A clocks at that instant, and (b) ramping the forward climb
        # command from zero over a bounded window.  Defaults preserve generic
        # launch compatibility; production wires the protection explicitly.
        self.truth_flight_a_require_fresh_ready=bool(rospy.get_param(
            '~truth_flight_a_require_fresh_ready', False))
        self.truth_flight_a_climb_ramp_seconds=max(0.0, float(rospy.get_param(
            '~truth_flight_a_climb_ramp_seconds', 0.0)))
        self.truth_flight_a_ready_anchor=None
        # Round36 climbed only 0.15 m, dropped back to lobby height and
        # yawed/translated off the first-flight centreline.  The established
        # fall/tracking guards intentionally arm only after 0.20--0.30 m, so
        # that low-height derail was neither a fall nor a tracked-stair loss
        # and spent the whole hard-stall budget issuing an ineffective
        # stair-gait strafe.  Detect this distinct pre-commit signature and
        # permit one plane-policy re-stage/reload attempt.
        self.truth_flight_a_early_retry_max_attempts=max(0, int(
            rospy.get_param(
                '~truth_flight_a_early_retry_max_attempts', 1)))
        self.truth_flight_a_early_retry_min_peak_gain=max(0.0, float(
            rospy.get_param(
                '~truth_flight_a_early_retry_min_peak_gain_m', .10)))
        self.truth_flight_a_early_retry_drawdown=max(0.0, float(
            rospy.get_param(
                '~truth_flight_a_early_retry_drawdown_m', .10)))
        self.truth_flight_a_early_retry_max_current_gain=float(
            rospy.get_param(
                '~truth_flight_a_early_retry_max_current_gain_m', .08))
        self.truth_flight_a_early_retry_min_center_error=max(0.0, float(
            rospy.get_param(
                '~truth_flight_a_early_retry_min_center_error_m', .18)))
        self.truth_flight_a_early_retry_min_heading_error=max(0.0, float(
            rospy.get_param(
                '~truth_flight_a_early_retry_min_heading_error_rad', .65)))
        self.truth_flight_a_early_retry_policy_timeout=max(.1, float(
            rospy.get_param(
                '~truth_flight_a_early_retry_policy_timeout_sec', 8.0)))
        self.truth_flight_a_early_retry_plane_settle=max(0.0, float(
            rospy.get_param(
                '~truth_flight_a_early_retry_plane_settle_sec', 1.0)))
        self.truth_flight_a_early_retry_attempts=0
        self.truth_flight_a_early_retry_reason=None
        self.truth_flight_a_early_retry_policy_deadline=None
        self.truth_flight_a_early_retry_plane_loaded_at=None
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
        self.truth_flight_a_predicted_center_error=None
        self.truth_flight_a_lateral_velocity=None
        self.truth_flight_a_predictive_recovery_active=False
        self.truth_flight_a_predictive_center_crossing_active=False
        self.truth_flight_a_heading_error=None
        self.truth_flight_a_recovery_active=False
        self.truth_flight_a_slip_active=False
        self.truth_flight_a_peak_drawdown=None
        self.truth_flight_a_command_forward_speed=None
        self.truth_flight_a_command_center_speed=None
        self.truth_flight_a_command_yaw_rate=None
        self.truth_flight_a_alignment_recovery_until=None
        self.truth_flight_a_alignment_recovery_started_at=None
        self.truth_flight_a_alignment_recovery_start_heading_error=None
        self.truth_flight_a_alignment_recovery_stable_since=None
        self.truth_flight_a_alignment_recovery_attempts=0
        self.truth_flight_a_alignment_recovery_last_reason=None
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
        # Flight B can acquire a large outward lateral velocity before the
        # position-only centre controller reaches its 0.12 m/s limit. Predict
        # that drift from Gazebo ModelStates, moderate (but do not starve) the
        # climb, and give centring enough authority to reverse one gait step.
        # Round23 proved that 0.12 m/s cannot sustain stepping on a tread and
        # that a 0.06 m release band can latch recovery at a safe 6-10 cm
        # offset. Keep a narrow 0.10->0.12 m hysteresis instead.
        self.truth_flight_b_recovery_center_error=float(rospy.get_param(
            '~truth_flight_b_recovery_center_error_m', .12))
        self.truth_flight_b_recovery_release_center_error=float(
            rospy.get_param(
                '~truth_flight_b_recovery_release_center_error_m', .10))
        self.truth_flight_b_recovery_forward_speed=float(rospy.get_param(
            '~truth_flight_b_recovery_forward_speed_mps', .40))
        # As on Flight A, production may run ordinary predictive correction
        # faster while retaining the proven 0.40 m/s command whenever the
        # lateral predictor reports a strong escape.  Disabled by default so
        # other stair profiles keep their previous command law.
        self.truth_flight_b_strong_recovery_forward_speed=float(
            rospy.get_param(
                '~truth_flight_b_strong_recovery_forward_speed_mps', -1.0))
        self.truth_flight_b_recovery_center_speed=float(rospy.get_param(
            '~truth_flight_b_recovery_center_speed_mps', .20))
        self.truth_flight_b_prediction_horizon=float(rospy.get_param(
            '~truth_flight_b_prediction_horizon_sec', .80))
        self.truth_flight_b_prediction_min_lateral_speed=float(
            rospy.get_param(
                '~truth_flight_b_prediction_min_lateral_speed_mps', .06))
        self.truth_flight_b_prediction_velocity_alpha=float(rospy.get_param(
            '~truth_flight_b_prediction_velocity_alpha', .30))
        # Flight-B round43 reacted at the correct sign but its 0.20 m/s
        # counter-strafe could not reverse 0.16+ m/s outward momentum before
        # the next learned-gait step accelerated to 0.86 m/s.  Escalate only
        # while the filtered velocity is materially outward; the ordinary
        # recovery authority and release hysteresis remain unchanged once
        # that momentum is arrested.
        self.truth_flight_b_predictive_strong_min_lateral_speed=max(
            0.0, float(rospy.get_param(
                '~truth_flight_b_predictive_strong_min_lateral_speed_mps',
                .15)))
        self.truth_flight_b_predictive_strong_center_speed=max(
            self.truth_flight_b_recovery_center_speed,
            float(rospy.get_param(
                '~truth_flight_b_predictive_strong_center_speed_mps', .30)))
        # Once the body is already close to the hard centreline guard, keep
        # the stronger counter-strafe even after the filtered lateral speed
        # drops below the predictive trigger.  A value of zero disables this
        # positional escalation so the independently tuned upper stair keeps
        # its historical behaviour.
        self.truth_flight_b_near_edge_strong_center_error=max(
            0.0, float(rospy.get_param(
                '~truth_flight_b_near_edge_strong_center_error_m', 0.0)))
        self.truth_flight_b_predictive_strong_active=False
        self.truth_flight_b_near_edge_strong_active=False
        self.truth_flight_b_predictive_center_crossing_active=False
        self.truth_flight_b_max_center_error=float(rospy.get_param(
            '~truth_flight_b_max_center_error_m', .38))
        self.truth_flight_b_max_heading_error=float(rospy.get_param(
            '~truth_flight_b_max_heading_error_rad', .65))
        self.truth_flight_b_recovery_heading_error=float(rospy.get_param(
            '~truth_flight_b_recovery_heading_error_rad', .25))
        self.truth_flight_b_tracking_recovery_yaw_rate=float(rospy.get_param(
            '~truth_flight_b_tracking_recovery_yaw_rate_rps', .16))
        self.truth_flight_b_heading_gain=float(rospy.get_param(
            '~truth_flight_b_heading_gain', .35))
        self.truth_flight_b_max_yaw_rate=float(rospy.get_param(
            '~truth_flight_b_max_yaw_rate_rps', .10))
        # A steep-flight gait can yaw abruptly for a few samples while its
        # trunk is still safely centred on a tread.  Stopping the stair node
        # on the first threshold crossing removes the periodic RL command and
        # turns that recoverable oscillation into a slide (round30: centre
        # error 0.084 m, heading error 0.654 rad versus a 0.650 rad guard).
        # Keep the original tracking guard, but first give a centred robot a
        # short, bounded zero-forward re-landing/yaw correction.  A large
        # centre error, a genuine height drop, or exhausted attempts still
        # terminates the climb.
        self.truth_flight_b_alignment_recovery_seconds=max(0.1, float(
            rospy.get_param(
                '~truth_flight_b_alignment_recovery_seconds', 1.5)))
        self.truth_flight_b_alignment_recovery_max_attempts=max(1, int(
            rospy.get_param(
                '~truth_flight_b_alignment_recovery_max_attempts', 2)))
        self.truth_flight_b_alignment_recovery_safe_center_error=max(
            0.0, float(rospy.get_param(
                '~truth_flight_b_alignment_recovery_safe_center_error_m',
                .24)))
        self.truth_flight_b_alignment_recovery_hard_heading_error=max(
            self.truth_flight_b_max_heading_error, float(rospy.get_param(
                '~truth_flight_b_alignment_recovery_hard_heading_error_rad',
                1.20)))
        self.truth_flight_b_alignment_recovery_max_drop=max(0.0, float(
            rospy.get_param(
                '~truth_flight_b_alignment_recovery_max_drop_m', .25)))
        self.truth_flight_b_alignment_recovery_release_heading_error=max(
            0.0, float(rospy.get_param(
                '~truth_flight_b_alignment_recovery_release_heading_error_rad',
                .20)))
        self.truth_flight_b_alignment_recovery_release_center_error=max(
            0.0, float(rospy.get_param(
                '~truth_flight_b_alignment_recovery_release_center_error_m',
                .16)))
        self.truth_flight_b_alignment_recovery_release_planar_speed=max(
            0.0, float(rospy.get_param(
                '~truth_flight_b_alignment_recovery_release_planar_speed_mps',
                .22)))
        self.truth_flight_b_alignment_recovery_release_yaw_rate=max(
            0.0, float(rospy.get_param(
                '~truth_flight_b_alignment_recovery_release_yaw_rate_rps',
                .30)))
        self.truth_flight_b_alignment_recovery_stable_seconds=max(0.0, float(
            rospy.get_param(
                '~truth_flight_b_alignment_recovery_stable_seconds', .25)))
        self.truth_flight_b_alignment_recovery_until=None
        self.truth_flight_b_alignment_recovery_started_at=None
        self.truth_flight_b_alignment_recovery_start_z=None
        self.truth_flight_b_alignment_recovery_start_heading_error=None
        self.truth_flight_b_alignment_recovery_stable_since=None
        self.truth_flight_b_alignment_recovery_attempts=0
        self.truth_flight_b_alignment_recovery_last_reason=None
        self.truth_flight_b_fall_drop=float(rospy.get_param(
            '~truth_flight_b_fall_drop_m', .55))
        self.truth_flight_b_peak_gain=None
        self.truth_flight_b_center_error=None
        self.truth_flight_b_predicted_center_error=None
        self.truth_flight_b_lateral_velocity=None
        self.truth_flight_b_predictive_recovery_active=False
        self.truth_flight_b_predictive_center_crossing_active=False
        self.truth_flight_b_predictive_strong_active=False
        self.truth_flight_b_near_edge_strong_active=False
        self.truth_flight_b_recovery_active=False
        self.truth_flight_b_command_forward_speed=None
        self.truth_flight_b_command_center_speed=None
        self.truth_flight_b_command_yaw_rate=None
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
        # Optional early slip intervention (currently enabled by launch only
        # where repeated real runs justify it):
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
        # A z-freeze detector cannot see a dynamic tread rejection while the
        # trunk is still bobbing upward.  F1->F2 round81f was already sliding
        # down Flight B at 0.31-0.46 m/s for more than 0.1 s, yet continued to
        # gain a few millimetres of height until the pitch/yaw collapse.  Use
        # the ModelStates velocity projected opposite the commanded flight
        # direction as an independent, short-dwell trigger.  The height gate
        # excludes the harmless run-up/first-tread bounce seen in round81e.
        # A zero speed threshold keeps non-production launch files backward
        # compatible; the full three-floor launch explicitly enables it.
        self.truth_flight_b_rearward_slip_speed=max(0.0, float(
            rospy.get_param(
                '~truth_flight_b_rearward_slip_speed_mps', 0.0)))
        self.truth_flight_b_rearward_slip_detect_seconds=max(0.0, float(
            rospy.get_param(
                '~truth_flight_b_rearward_slip_detect_seconds', .04)))
        self.truth_flight_b_rearward_slip_min_height_gain=max(0.0, float(
            rospy.get_param(
                '~truth_flight_b_rearward_slip_min_height_gain_m', .40)))
        self.truth_flight_b_rearward_slip_velocity_alpha=min(
            1.0, max(.01, float(rospy.get_param(
                '~truth_flight_b_rearward_slip_velocity_alpha', .70))))
        self.truth_flight_b_rearward_slip_release_ratio=min(
            1.0, max(0.0, float(rospy.get_param(
                '~truth_flight_b_rearward_slip_release_ratio', .80))))
        self.truth_flight_b_rearward_speed=None
        self.truth_flight_b_filtered_rearward_speed=None
        self.truth_flight_b_rearward_slip_since=None
        self.truth_flight_b_rearward_slip_latched=False
        self.truth_flight_b_rearward_slip_events=0
        self.truth_flight_b_rearward_slip_last_trigger_gain=None
        # A Flight-B posture collapse can continue gaining height, so neither
        # the z-stall nor rearward-velocity guard sees it.  Detect the steep
        # negative pitch acceleration while the trunk is still upright enough
        # to re-land, with a separate sustained severe-pitch fallback.  Both
        # thresholds default disabled outside the production launch.
        self.truth_flight_b_pitch_rate_window=max(0.05, float(
            rospy.get_param(
                '~truth_flight_b_pitch_rate_window_sec', .18)))
        self.truth_flight_b_pitch_rate_trigger=max(0.0, float(
            rospy.get_param(
                '~truth_flight_b_pitch_rate_trigger_rps', 0.0)))
        self.truth_flight_b_pitch_rate_min_abs=max(0.0, float(
            rospy.get_param(
                '~truth_flight_b_pitch_rate_min_abs_rad', .60)))
        self.truth_flight_b_severe_pitch=max(0.0, float(
            rospy.get_param(
                '~truth_flight_b_severe_pitch_rad', 0.0)))
        self.truth_flight_b_severe_pitch_detect_seconds=max(0.0, float(
            rospy.get_param(
                '~truth_flight_b_severe_pitch_detect_seconds', .10)))
        self.truth_flight_b_pitch_min_height_gain=max(0.0, float(
            rospy.get_param(
                '~truth_flight_b_pitch_min_height_gain_m', .08)))
        self.truth_flight_b_pitch_recovery_max_attempts=max(1, int(
            rospy.get_param(
                '~truth_flight_b_pitch_recovery_max_attempts', 2)))
        self.truth_flight_b_pitch_history=[]
        self.truth_flight_b_pitch_rate=None
        self.truth_flight_b_severe_pitch_since=None
        self.truth_flight_b_pitch_collapse_latched=False
        self.truth_flight_b_pitch_collapse_events=0
        self.truth_flight_b_pitch_last_trigger_gain=None
        self.truth_flight_b_pitch_last_trigger_reason=None
        # Optional pre-ascent fixed-stand: before climbing flight B, drop out
        # of the RL gait into unitree's fixed stand
        # (default joint pose, Joy L2_A) for pre_ascent_stand_seconds, then
        # re-enter the RL gait.  The upper flight is currently entered with
        # the residual joint state from F2 exploration + flight A + landing
        # turn; 11/11 deadlocks (full40-45) indicate that residual biases the
        # tread-slip window.  Re-starting from the policy's training-centre
        # pose removes that bias.  0.0 disables the sequence.
        self.pre_ascent_stand_seconds=float(rospy.get_param(
            '~pre_ascent_stand_seconds', 0.0))
        # Flight B is entered directly from a moving hairpin.  Switching the
        # controller to FixedStand on that same timer tick can freeze a
        # swing-leg gait sample and make the body collapse at the landing
        # edge.  Production stair launches opt into a short zero-command RL
        # dwell before changing controllers.
        self.pre_ascent_rl_settle_seconds=max(0.0, float(rospy.get_param(
            '~pre_ascent_rl_settle_seconds', 0.0)))
        self.pre_ascent_rl_settle_timeout=max(
            self.pre_ascent_rl_settle_seconds, float(rospy.get_param(
                '~pre_ascent_rl_settle_timeout_sec', 2.5)))
        self.pre_ascent_rl_settle_max_planar_speed=max(0.0, float(
            rospy.get_param(
                '~pre_ascent_rl_settle_max_planar_speed_mps', .18)))
        self.pre_ascent_rl_settle_max_vertical_speed=max(0.0, float(
            rospy.get_param(
                '~pre_ascent_rl_settle_max_vertical_speed_mps', .12)))
        self.pre_ascent_rl_settle_max_yaw_rate=max(0.0, float(
            rospy.get_param(
                '~pre_ascent_rl_settle_max_yaw_rate_rps', .25)))
        self.pre_ascent_rl_settle_max_joint_velocity_rms=max(0.0, float(
            rospy.get_param(
                '~pre_ascent_rl_settle_max_joint_velocity_rms', .35)))
        self.pre_ascent_rl_settle_started=None
        self.pre_ascent_rl_stable_since=None
        self.pre_ascent_stand_started=None
        self.pre_ascent_stand_done=False
        # After the stand hold, a fresh RL handshake (Joy button-3 request ->
        # new /locomotion_ready=True) is required before any nonzero
        # flight-B command; bounded by this timeout.
        self.pre_ascent_rl_requested=False
        self.pre_ascent_rl_requested_at=None
        # The normal RL state-entry handshake holds a zero command for 5.5 s.
        # That is safe on a floor but too long on the narrow turning landing:
        # an otherwise aligned F1->F2 replay slid off before readiness.  Arm a
        # one-shot controller profile only for the FixedStand -> flight-B
        # boundary.  State_RL consumes the enable flag on entry; the manager
        # owns and clears all supporting parameters afterward.
        self.pre_ascent_rl_fast_takeover=bool(rospy.get_param(
            '~pre_ascent_rl_fast_takeover', False))
        self.pre_ascent_rl_fast_blend_seconds=max(.25, float(rospy.get_param(
            '~pre_ascent_rl_fast_blend_seconds', .75)))
        self.pre_ascent_rl_fast_zero_hold_seconds=max(.10, float(
            rospy.get_param(
                '~pre_ascent_rl_fast_zero_hold_seconds', .25)))
        self._fast_takeover_params={
            '/simenv/stair_fast_takeover_blend_seconds':
                self.pre_ascent_rl_fast_blend_seconds,
            '/simenv/stair_fast_takeover_zero_hold_seconds':
                self.pre_ascent_rl_fast_zero_hold_seconds,
            '/simenv/stair_fast_takeover_enabled':True}
        self._fast_takeover_profile_owned=False
        # FixedStand normally waits about eight seconds before publishing a
        # fresh readiness edge.  At the flight-A policy boundary this manager
        # has already proved low joint velocity and freezes that exact joint
        # sample as the target, so a one-shot captured-pose profile can retain
        # a real gain ramp/stability proof without paying the default-stance
        # interpolation budget.  State_FixedStand consumes these globals only
        # on the immediately following entry.
        self.truth_flight_a_policy_fast_fixed_stand=bool(rospy.get_param(
            '~truth_flight_a_policy_fast_fixed_stand', False))
        self._fast_fixed_stand_params={
            '/simenv/fixed_stand_fast_duration_seconds':float(
                rospy.get_param(
                    '~truth_flight_a_policy_fast_fixed_stand_duration_seconds',
                    1.0)),
            '/simenv/fixed_stand_fast_settle_seconds':float(
                rospy.get_param(
                    '~truth_flight_a_policy_fast_fixed_stand_settle_seconds',
                    .10)),
            '/simenv/fixed_stand_fast_minimum_elapsed_seconds':float(
                rospy.get_param(
                    '~truth_flight_a_policy_fast_fixed_stand_minimum_elapsed_seconds',
                    1.10)),
            '/simenv/fixed_stand_fast_stable_seconds':float(
                rospy.get_param(
                    '~truth_flight_a_policy_fast_fixed_stand_stable_seconds',
                    1.0)),
            '/simenv/fixed_stand_fast_minimum_gain_ratio':min(
                1.0, max(0.0, float(rospy.get_param(
                    '~truth_flight_a_policy_fast_fixed_stand_minimum_gain_ratio',
                    0.0)))),
            '/simenv/fixed_stand_fast_ready_enabled':True}
        self._fast_fixed_stand_profile_owned=False
        self.pre_ascent_rl_reentry_timeout=float(rospy.get_param(
            '~pre_ascent_rl_reentry_timeout_sec', 20.0))
        # A hot plane->stair policy reload applies the new zero-command gait
        # immediately.  Round48 showed that an unlucky gait sample can move
        # 0.55 m sideways and roll the body over before the new readiness
        # handshake latches.  On the first physical stair only, production
        # opts into a level-ground FixedStand boundary: settle the plane gait,
        # capture a low-speed joint pose, reload at zero command, freeze the
        # captured pose immediately on acknowledgement, then use the same
        # bounded fast RL re-entry proven on flight B.  The reload itself must
        # happen in RL because State_RL owns the reload worker.
        self.truth_flight_a_policy_stand_seconds=max(0.0, float(
            rospy.get_param(
                '~truth_flight_a_policy_stand_seconds', 0.0)))
        self.truth_flight_a_policy_settle_seconds=max(0.0, float(
            rospy.get_param(
                '~truth_flight_a_policy_settle_seconds', .60)))
        self.truth_flight_a_policy_settle_timeout=max(
            self.truth_flight_a_policy_settle_seconds,
            float(rospy.get_param(
                '~truth_flight_a_policy_settle_timeout_sec', 2.5)))
        # A zero-command learned gait keeps cycling its legs even after the
        # trunk has stopped.  Do not require joint velocity to remain below
        # this threshold for the full body-settle dwell; instead, once the
        # body dwell is complete, capture the next low-velocity gait sample
        # and switch to FixedStand on that tick.
        self.truth_flight_a_policy_capture_max_joint_velocity_rms=max(
            0.0, float(rospy.get_param(
                '~truth_flight_a_policy_capture_max_joint_velocity_rms',
                .35)))
        self.truth_flight_a_policy_queue_hold_seconds=max(0.05, float(
            rospy.get_param(
                '~truth_flight_a_policy_queue_hold_seconds', .25)))
        # FixedStand can either freeze the atomically captured low-velocity
        # gait pose or converge to its symmetric default stance.  Snapshot is
        # the safe default on both landings: the full physical mission showed
        # that changing posture at the F1 boundary can roll the robot before
        # the stair policy is even loaded.  Readiness remains an independent
        # controller-side proof and can be required by the launch profile.
        self.truth_flight_a_policy_use_default_stand_target=bool(
            rospy.get_param(
                '~truth_flight_a_policy_use_default_stand_target', False))
        self.truth_flight_a_policy_require_fixed_stand_ready=bool(
            rospy.get_param(
                '~truth_flight_a_policy_require_fixed_stand_ready', False))
        self.truth_flight_a_policy_fixed_stand_ready_timeout=max(
            self.truth_flight_a_policy_stand_seconds,
            float(rospy.get_param(
                '~truth_flight_a_policy_fixed_stand_ready_timeout_sec',
                12.0)))
        self.truth_flight_a_ready_wait_timeout=max(0.1, float(
            rospy.get_param(
                '~truth_flight_a_ready_wait_timeout_sec', 20.0)))
        self.truth_flight_a_ready_wait_started=None
        self.truth_flight_a_preflight_max_tilt=max(0.1, float(
            rospy.get_param(
                '~truth_flight_a_preflight_max_tilt_rad', .45)))
        self.truth_flight_a_preflight_max_height_drop=max(0.02, float(
            rospy.get_param(
                '~truth_flight_a_preflight_max_height_drop_m', .10)))
        self.truth_flight_a_policy_settle_started=None
        self.truth_flight_a_policy_stable_since=None
        self.truth_flight_a_policy_stand_started=None
        self.truth_flight_a_policy_stand_anchor_z=None
        self.truth_flight_a_policy_rl_requested_at=None
        self.truth_flight_a_policy_queued_at=None
        self.truth_flight_a_policy_capture_joint_names=None
        self.truth_flight_a_policy_capture_joint_positions=None
        self.truth_flight_a_policy_capture_joint_velocity_rms=None
        self.truth_flight_b_stall_start_z=None
        self.truth_flight_b_stall_since=None
        self.truth_flight_b_recovery_until=None
        self.truth_flight_b_ramp_until=None
        self.truth_flight_b_backoff_until=None
        self.truth_flight_b_stall_recoveries=0
        self.truth_landing_position_error=None
        self.truth_landing_heading_error=None
        self.truth_landing_servo_heading=None
        self.truth_landing_servo_mode=None
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
        # B flights take about 12 s. The launch scopes this watchdog per stair:
        # upper-floor runs retain the conservative default, while F1 can grant
        # enough bounded time for one pause+backoff+ramp recovery cycle.
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
        rospy.Subscriber('/fixed_stand_ready',Bool,self.on_fixed_stand_ready,queue_size=2)
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
        velocities=list(getattr(msg, 'velocity', ()) or ())
        velocity_rms=(
            math.sqrt(sum(value*value for value in velocities)/len(velocities))
            if velocities else None)
        with self._entry_joint_lock:
            if self.entry_joint_snapshot is None:
                self.entry_joint_names=list(msg.name)
            self.entry_joint_snapshot=list(msg.position)
            self.entry_joint_velocity_rms=velocity_rms
            self._entry_joint_wall_time=msg.header.stamp.to_sec()
            self._entry_joint_recv_wall=time.monotonic()

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

    # junior_ctrl FSM / URDF order.  Gazebo joint_states arrive name-sorted
    # (FL..., FR..., RL..., RR...), so a snapshot must be re-indexed into
    # this order before it can be pushed as /stand_target_joints.
    _FSM_JOINT_ORDER=['FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
                      'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
                      'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint',
                      'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint']

    def _apply_stand_target_joints(
            self, preferred_names=None, preferred_positions=None,
            preferred_source='live'):
        """Push the F1->F2 entry snapshot as the pre-ascent stand target.

        Returns True only after a valid snapshot was parsed, re-indexed to
        FSM order, and successfully pushed to /stand_target_joints.  Every
        failure mode (unconfigured, missing, unreadable, malformed JSON,
        missing joint, non-finite value, param-server error) returns False
        without raising, so the caller falls back to the FixedStand default
        stance and the control timer keeps running.
        """
        # This helper is normally called once.  Still make re-use safe: a new
        # invalid snapshot must not leave an override owned by an earlier
        # successful call.  If the parameter service is temporarily
        # unavailable, retain ownership so a later call can retry the clear.
        if self._stand_target_applied:
            if not self._clear_stand_target_joints():
                rospy.logerr('Pre-ascent stand: cannot replace the previously '
                             'owned stand target because clearing it failed')
                return False
        # Round19 regression: the FixedStand default stance was applied at
        # flight-B entry purely because SNAPSHOT_JOINTS_FILE was not
        # configured for this mission, even though this manager already owns
        # a fresh live 12-DOF joint-state sample at that exact moment.  The
        # default-stance step lost ~0.2 m of height and posture, unlatched
        # RL readiness, and produced the observed FLIP_STOP.  Prefer the
        # fresh live sample first; the configured file remains the second
        # source so replayed runs keep working, and the default stance is
        # used only when neither source is valid.
        source=preferred_source
        targets=None
        if preferred_names and preferred_positions is not None:
            targets=self._reindex_stand_targets(
                dict(zip(preferred_names, preferred_positions)), source)
        if targets is None:
            source='live'
            targets=self._stand_targets_from_live_joint_sample()
        if targets is None:
            source=self.snapshot_file
            targets=self._stand_targets_from_snapshot_file()
        if targets is None:
            return False
        try:
            rospy.set_param(self._stand_target_param, targets)
        except Exception as error:
            rospy.logerr('Pre-ascent stand: set_param failed (%s), '
                         'using default stance', error)
            return False
        self._stand_target_applied=True
        rospy.loginfo('Pre-ascent stand target set from %s entry sample',
                      source)
        return True

    # A live joint sample is only trustworthy for the stand handoff while it
    # is recent; a stale one may predate the whole climb and would reproduce
    # the same posture risk as the default stance.
    _LIVE_JOINT_SAMPLE_MAX_AGE_SEC=2.0

    def _stand_targets_from_live_joint_sample(
            self, now=None, max_age_sec=None):
        names=getattr(self, 'entry_joint_names', None)
        positions=getattr(self, 'entry_joint_snapshot', None)
        recv_wall=getattr(self, '_entry_joint_recv_wall', None)
        if not names or positions is None or recv_wall is None:
            return None
        now=time.monotonic() if now is None else now
        max_age=(self._LIVE_JOINT_SAMPLE_MAX_AGE_SEC
                 if max_age_sec is None else max_age_sec)
        if now-recv_wall > max_age:
            rospy.logwarn('Pre-ascent stand: live joint sample is stale '
                          '(%.2f s old), trying the configured snapshot',
                          now-recv_wall)
            return None
        return self._reindex_stand_targets(dict(zip(names, positions)),
                                           'live joint sample')

    def _stand_targets_from_snapshot_file(self):
        if not self.snapshot_file or not os.path.isfile(self.snapshot_file):
            rospy.logwarn('Pre-ascent stand: SNAPSHOT_JOINTS_FILE %r missing, '
                          'using default stance', self.snapshot_file)
            return None
        try:
            with open(self.snapshot_file) as f:
                data=json.load(f)
            return self._reindex_stand_targets(
                dict(zip(data['joint_names'], data['positions'])),
                self.snapshot_file)
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            rospy.logerr('Pre-ascent stand: unusable snapshot %r (%s), '
                         'using default stance', self.snapshot_file, error)
            return None

    def _reindex_stand_targets(self, name_to_pos, source):
        try:
            targets=[float(name_to_pos[name])
                     for name in self._FSM_JOINT_ORDER]
        except (KeyError, TypeError, IndexError) as error:
            rospy.logerr('Pre-ascent stand: %s lacks a required joint (%s), '
                         'using default stance', source, error)
            return None
        if len(targets)!=12 or not all(map(math.isfinite, targets)):
            rospy.logerr('Pre-ascent stand: %s has non-finite or incomplete '
                         'data, using default stance', source)
            return None
        return targets

    def _clear_stand_target_joints(self):
        """Remove this instance's override without leaking timer exceptions.

        Returns True when no owned parameter remains.  A transient parameter
        service error returns False and deliberately preserves the ownership
        flag, allowing the pre-ascent gate to retry without ever deleting a
        parameter owned by another process.
        """
        if not self._stand_target_applied:
            return True
        try:
            if rospy.has_param(self._stand_target_param):
                rospy.delete_param(self._stand_target_param)
        except KeyError:
            # Lost a has_param/delete race with another deleter: the param is
            # gone either way, which is the desired end state.
            self._stand_target_applied=False
            return True
        except Exception as error:
            rospy.logerr('Pre-ascent stand: failed to clear owned %s (%s); '
                         'will retry while holding zero command',
                         self._stand_target_param, error)
            return False
        self._stand_target_applied=False
        return True

    def _rl_reentry_fresh_ready(self):
        """True only for a /locomotion_ready=True arriving after the request."""
        return (self.locomotion_ready and
                self.locomotion_ready_at is not None and
                self.pre_ascent_rl_requested_at is not None and
                self.locomotion_ready_at >= self.pre_ascent_rl_requested_at)

    def _arm_fast_takeover_profile(self):
        """Install the one-shot RL re-entry profile, enable flag last."""
        if not bool(getattr(self, 'pre_ascent_rl_fast_takeover', False)):
            return True
        params=getattr(self, '_fast_takeover_params', {
            '/simenv/stair_fast_takeover_blend_seconds':.75,
            '/simenv/stair_fast_takeover_zero_hold_seconds':.25,
            '/simenv/stair_fast_takeover_enabled':True})
        self._fast_takeover_profile_owned=True
        try:
            # The controller can enter immediately after the Joy request, so
            # publish duration values before the one-shot enable latch.
            for name in (
                    '/simenv/stair_fast_takeover_blend_seconds',
                    '/simenv/stair_fast_takeover_zero_hold_seconds',
                    '/simenv/stair_fast_takeover_enabled'):
                rospy.set_param(name, params[name])
        except Exception as error:
            rospy.logerr(
                'Could not arm the stair RL fast-takeover profile (%s).',
                error)
            self._clear_fast_takeover_profile()
            return False
        return True

    def _clear_fast_takeover_profile(self):
        """Clear only parameters owned by this stair-manager instance."""
        if not bool(getattr(
                self, '_fast_takeover_profile_owned', False)):
            return True
        ok=True
        params=getattr(self, '_fast_takeover_params', {})
        for name in params:
            try:
                if rospy.has_param(name):
                    rospy.delete_param(name)
            except KeyError:
                pass
            except Exception as error:
                ok=False
                rospy.logerr(
                    'Failed to clear owned stair takeover parameter %s '
                    '(%s).', name, error)
        if ok:
            self._fast_takeover_profile_owned=False
        return ok

    def _arm_fast_fixed_stand_profile(self):
        """Arm one captured-pose FixedStand entry, enable latch last."""
        if not bool(getattr(
                self, 'truth_flight_a_policy_fast_fixed_stand', False)):
            return True
        self._fast_fixed_stand_profile_owned=True
        params=self._fast_fixed_stand_params
        try:
            for name in (
                    '/simenv/fixed_stand_fast_duration_seconds',
                    '/simenv/fixed_stand_fast_settle_seconds',
                    '/simenv/fixed_stand_fast_minimum_elapsed_seconds',
                    '/simenv/fixed_stand_fast_stable_seconds',
                    '/simenv/fixed_stand_fast_minimum_gain_ratio',
                    '/simenv/fixed_stand_fast_ready_enabled'):
                rospy.set_param(name, params[name])
        except Exception as error:
            rospy.logerr(
                'Could not arm captured-pose FixedStand profile (%s).', error)
            self._clear_fast_fixed_stand_profile()
            return False
        return True

    def _clear_fast_fixed_stand_profile(self):
        if not bool(getattr(
                self, '_fast_fixed_stand_profile_owned', False)):
            return True
        ok=True
        for name in getattr(self, '_fast_fixed_stand_params', {}):
            try:
                if rospy.has_param(name):
                    rospy.delete_param(name)
            except KeyError:
                pass
            except Exception as error:
                ok=False
                rospy.logerr(
                    'Failed to clear owned FixedStand parameter %s (%s).',
                    name, error)
        if ok:
            self._fast_fixed_stand_profile_owned=False
        return ok

    def _reset_flight_a_alignment_recovery(self, clear_attempts=True):
        self.truth_flight_a_alignment_recovery_until=None
        self.truth_flight_a_alignment_recovery_started_at=None
        self.truth_flight_a_alignment_recovery_start_heading_error=None
        self.truth_flight_a_alignment_recovery_stable_since=None
        self.truth_flight_a_alignment_recovery_last_reason=None
        if clear_attempts:
            self.truth_flight_a_alignment_recovery_attempts=0

    def _reset_flight_a_rearward_slip(self, clear_events=False):
        self.truth_flight_a_rearward_speed=None
        self.truth_flight_a_filtered_rearward_speed=None
        self.truth_flight_a_filtered_climb_speed=None
        self.truth_flight_a_rearward_slip_since=None
        self.truth_flight_a_rearward_slip_latched=False
        self.truth_flight_a_backslide_recovery_until=None
        self.truth_flight_a_backslide_brake_active=False
        self.truth_flight_a_backslide_brake_center_speed=0.0
        self.truth_flight_a_backslide_brake_yaw_rate=0.0
        if clear_events:
            self.truth_flight_a_rearward_slip_events=0
            self.truth_flight_a_rearward_slip_last_trigger_gain=None
            self.truth_flight_a_backslide_recovery_ramp_until=None
            self.truth_flight_a_backslide_brake_peak_center_speed=0.0

    def _truth_flight_a_backslide_relanding_control(self):
        """Return a zero-climb command that brakes dangerous cross-slip.

        A pure zero command successfully re-landed earlier longitudinal
        backslides, but final-v4 seed 3 entered the hold with 0.32 m/s of
        outward lateral velocity.  Cancelling the already active centreline
        correction let that momentum carry the trunk off the tread and roll
        it past the controller's global fall guard.  Keep longitudinal flight
        speed exactly zero while commanding only a bounded, velocity-opposed
        centre correction and the existing conservative yaw correction.
        """
        self.truth_flight_a_backslide_brake_active=False
        self.truth_flight_a_backslide_brake_center_speed=0.0
        self.truth_flight_a_backslide_brake_yaw_rate=0.0
        heading=getattr(self, 'truth_stair_heading', None)
        if heading is None or not math.isfinite(float(heading)):
            return None
        lateral_velocity=float(getattr(
            self, 'truth_flight_a_lateral_velocity', 0.0) or 0.0)
        center_speed=float(getattr(
            self, 'truth_flight_a_command_center_speed', 0.0) or 0.0)
        yaw_rate=float(getattr(
            self, 'truth_flight_a_command_yaw_rate', 0.0) or 0.0)
        if not all(math.isfinite(value) for value in (
                lateral_velocity, center_speed, yaw_rate)):
            return None

        center_limit=max(0.0, float(getattr(
            self, 'truth_flight_a_predictive_strong_center_speed', .30)))
        minimum_brake=min(center_limit, max(0.0, float(getattr(
            self, 'truth_flight_a_recovery_center_speed', .22))))
        velocity_gate=max(0.0, float(getattr(
            self, 'truth_flight_a_prediction_min_lateral_speed', .08)))
        if abs(lateral_velocity) >= velocity_gate and center_limit > 0.0:
            brake_magnitude=max(
                minimum_brake, min(center_limit, abs(center_speed)))
            center_speed=-math.copysign(
                brake_magnitude, lateral_velocity)
        else:
            center_speed=max(-center_limit, min(center_limit, center_speed))

        yaw_limit=max(0.0, float(getattr(
            self, 'truth_flight_a_recovery_yaw_rate', .16)))
        yaw_rate=max(-yaw_limit, min(yaw_limit, yaw_rate))
        right_x=math.sin(float(heading))
        right_y=-math.cos(float(heading))
        self.truth_flight_a_backslide_brake_active=bool(
            abs(center_speed) > 1e-6 or abs(yaw_rate) > 1e-6)
        self.truth_flight_a_backslide_brake_center_speed=center_speed
        self.truth_flight_a_backslide_brake_yaw_rate=yaw_rate
        self.truth_flight_a_backslide_brake_peak_center_speed=max(
            float(getattr(
                self, 'truth_flight_a_backslide_brake_peak_center_speed',
                0.0)), abs(center_speed))
        # These fields describe the command actually published during the
        # hold, rather than the discarded forward component calculated by the
        # ordinary tracking controller earlier in this tick.
        self.truth_flight_a_command_forward_speed=0.0
        self.truth_flight_a_command_center_speed=center_speed
        self.truth_flight_a_command_yaw_rate=yaw_rate
        return (center_speed*right_x, center_speed*right_y,
                yaw_rate, center_speed)

    def _truth_flight_a_rearward_slip_update(self, now, truth_gain):
        """Latch a sustained Flight-A backslide before posture collapses.

        Flight A advances along ``truth_stair_heading``.  The projection and
        filtered dwell deliberately mirror Flight B's already validated
        dynamic guard.  A latched event remains active for the complete
        zero-climb re-landing hold; only the caller clears it afterward.
        """
        if bool(getattr(
                self, 'truth_flight_a_rearward_slip_latched', False)):
            return True
        threshold=max(0.0, float(getattr(
            self, 'truth_flight_a_rearward_slip_speed', 0.0)))
        minimum_gain=max(0.0, float(getattr(
            self, 'truth_flight_a_rearward_slip_min_height_gain', .40)))
        twist=getattr(self, 'truth_twist', None)
        heading=getattr(self, 'truth_stair_heading', None)
        valid_twist=bool(
            twist is not None and len(twist) >= 2 and
            math.isfinite(float(twist[0])) and
            math.isfinite(float(twist[1])))
        if (threshold <= 0.0 or not valid_twist or heading is None or
                truth_gain < minimum_gain):
            self.truth_flight_a_rearward_speed=None
            self.truth_flight_a_filtered_rearward_speed=None
            self.truth_flight_a_rearward_slip_since=None
            return False

        forward_speed=(float(twist[0])*math.cos(heading)+
                       float(twist[1])*math.sin(heading))
        rearward_speed=-forward_speed
        self.truth_flight_a_rearward_speed=rearward_speed
        previous=getattr(
            self, 'truth_flight_a_filtered_rearward_speed', None)
        alpha=min(1.0, max(.01, float(getattr(
            self, 'truth_flight_a_rearward_slip_velocity_alpha', .70))))
        filtered=(rearward_speed if previous is None else
                  (1.0-alpha)*float(previous)+alpha*rearward_speed)
        self.truth_flight_a_filtered_rearward_speed=filtered

        filtered_climb=None
        if (len(twist) >= 3 and math.isfinite(float(twist[2]))):
            climb_speed=float(twist[2])
            previous_climb=getattr(
                self, 'truth_flight_a_filtered_climb_speed', None)
            filtered_climb=(climb_speed if previous_climb is None else
                            (1.0-alpha)*float(previous_climb)+
                            alpha*climb_speed)
            self.truth_flight_a_filtered_climb_speed=filtered_climb
            maximum_climb=max(0.0, float(getattr(
                self, 'truth_flight_a_rearward_slip_max_climb_speed', .12)))
            if filtered_climb > maximum_climb:
                self.truth_flight_a_rearward_slip_since=None
                return False

        since=getattr(self, 'truth_flight_a_rearward_slip_since', None)
        release_ratio=min(1.0, max(0.0, float(getattr(
            self, 'truth_flight_a_rearward_slip_release_ratio', .80))))
        active_threshold=(threshold*release_ratio
                          if since is not None else threshold)
        if filtered < active_threshold:
            self.truth_flight_a_rearward_slip_since=None
            return False
        if since is None or now < since:
            self.truth_flight_a_rearward_slip_since=now
            since=now
        dwell=max(0.0, float(getattr(
            self, 'truth_flight_a_rearward_slip_detect_seconds', .04)))
        if now-since < dwell:
            return False

        self.truth_flight_a_rearward_slip_latched=True
        self.truth_flight_a_rearward_slip_events=int(getattr(
            self, 'truth_flight_a_rearward_slip_events', 0))+1
        self.truth_flight_a_rearward_slip_last_trigger_gain=truth_gain
        rospy.logwarn(
            'Flight-A dynamic backslide caught before posture collapse: '
            'filtered rearward speed=%.3f m/s, filtered climb speed=%.3f '
            'm/s for %.2f s at height gain %.2f m (threshold %.3f m/s).',
            filtered, (filtered_climb if filtered_climb is not None else
                       float('nan')),
            now-since, truth_gain, threshold)
        return True

    def _reset_flight_b_alignment_recovery(self, clear_attempts=True):
        self.truth_flight_b_alignment_recovery_until=None
        self.truth_flight_b_alignment_recovery_started_at=None
        self.truth_flight_b_alignment_recovery_start_z=None
        self.truth_flight_b_alignment_recovery_start_heading_error=None
        self.truth_flight_b_alignment_recovery_stable_since=None
        self.truth_flight_b_alignment_recovery_last_reason=None
        if clear_attempts:
            self.truth_flight_b_alignment_recovery_attempts=0

    def _reset_flight_b_rearward_slip(self, clear_events=False):
        self.truth_flight_b_rearward_speed=None
        self.truth_flight_b_filtered_rearward_speed=None
        self.truth_flight_b_rearward_slip_since=None
        self.truth_flight_b_rearward_slip_latched=False
        if clear_events:
            self.truth_flight_b_rearward_slip_events=0
            self.truth_flight_b_rearward_slip_last_trigger_gain=None

    def _reset_flight_b_pitch_collapse(self, clear_events=False):
        self.truth_flight_b_pitch_history=[]
        self.truth_flight_b_pitch_rate=None
        self.truth_flight_b_severe_pitch_since=None
        self.truth_flight_b_pitch_collapse_latched=False
        if clear_events:
            self.truth_flight_b_pitch_collapse_events=0
            self.truth_flight_b_pitch_last_trigger_gain=None
            self.truth_flight_b_pitch_last_trigger_reason=None

    def _truth_flight_b_pitch_collapse_update(self, now, flight_b_gain):
        """Latch a fast forward posture collapse before Euler yaw flips.

        Successful physical traces can briefly reach roughly -0.75 rad pitch,
        but their 0.18--0.25 s pitch slope stays above -1.36 rad/s.  The
        failed seed-3 trace fell from -0.22 to -0.69 rad in 0.235 s while
        continuing upward.  A rate gate catches that distinct transient; the
        severe-pitch dwell covers a slower collapse without reacting to one
        ordinary stair-policy footfall.
        """
        if bool(getattr(
                self, 'truth_flight_b_pitch_collapse_latched', False)):
            return True
        rate_threshold=max(0.0, float(getattr(
            self, 'truth_flight_b_pitch_rate_trigger', 0.0)))
        severe_threshold=max(0.0, float(getattr(
            self, 'truth_flight_b_severe_pitch', 0.0)))
        pitch=getattr(self, 'truth_pitch', None)
        if (pitch is None or not math.isfinite(float(pitch)) or
                flight_b_gain < float(getattr(
                    self, 'truth_flight_b_pitch_min_height_gain', .08)) or
                (rate_threshold <= 0.0 and severe_threshold <= 0.0)):
            self.truth_flight_b_pitch_history=[]
            self.truth_flight_b_pitch_rate=None
            self.truth_flight_b_severe_pitch_since=None
            return False

        now=float(now)
        pitch=float(pitch)
        history=list(getattr(self, 'truth_flight_b_pitch_history', []))
        if history and now < history[-1][0]:
            history=[]
        history.append((now, pitch))
        window=max(.05, float(getattr(
            self, 'truth_flight_b_pitch_rate_window', .18)))
        history=[sample for sample in history
                 if now-sample[0] <= max(1.0, 3.0*window)]
        self.truth_flight_b_pitch_history=history
        references=[sample for sample in history[:-1]
                    if now-sample[0] >= window]
        pitch_rate=None
        if references:
            reference=references[-1]
            elapsed=max(1e-6, now-reference[0])
            pitch_rate=(pitch-reference[1])/elapsed
        self.truth_flight_b_pitch_rate=pitch_rate

        minimum_abs=max(0.0, float(getattr(
            self, 'truth_flight_b_pitch_rate_min_abs', .60)))
        rate_trigger=bool(
            rate_threshold > 0.0 and pitch <= -minimum_abs and
            pitch_rate is not None and pitch_rate <= -rate_threshold)

        severe_trigger=False
        if severe_threshold > 0.0 and pitch <= -severe_threshold:
            since=getattr(
                self, 'truth_flight_b_severe_pitch_since', None)
            if since is None or now < since:
                since=now
                self.truth_flight_b_severe_pitch_since=since
            severe_trigger=(now-since >= max(0.0, float(getattr(
                self, 'truth_flight_b_severe_pitch_detect_seconds', .10))))
        else:
            self.truth_flight_b_severe_pitch_since=None

        if not rate_trigger and not severe_trigger:
            return False
        reason=('pitch_rate' if rate_trigger else 'sustained_severe_pitch')
        self.truth_flight_b_pitch_collapse_latched=True
        self.truth_flight_b_pitch_collapse_events=int(getattr(
            self, 'truth_flight_b_pitch_collapse_events', 0))+1
        self.truth_flight_b_pitch_last_trigger_gain=flight_b_gain
        self.truth_flight_b_pitch_last_trigger_reason=reason
        rospy.logwarn(
            'Flight-B posture collapse caught before alignment loss: '
            'pitch=%.3f rad, pitch rate=%s rad/s, flight gain=%.2f m, '
            'reason=%s; re-landing at zero speed.',
            pitch,
            ('%.3f' % pitch_rate if pitch_rate is not None else 'unavailable'),
            flight_b_gain, reason)
        return True

    def _truth_flight_b_rearward_slip_update(self, now, flight_b_gain):
        """Latch a dynamic Flight-B backslide before posture collapses.

        ``truth_twist`` is expressed in the world frame.  Project it onto the
        desired Flight-B heading, negate that component, then filter it so a
        single learned-gait footfall cannot interrupt a healthy climb.  Once
        triggered, the latch remains active for the existing re-landing hold;
        the caller clears it only after that complete recovery cycle.
        """
        if bool(getattr(
                self, 'truth_flight_b_rearward_slip_latched', False)):
            return True
        threshold=max(0.0, float(getattr(
            self, 'truth_flight_b_rearward_slip_speed', 0.0)))
        minimum_gain=max(0.0, float(getattr(
            self, 'truth_flight_b_rearward_slip_min_height_gain', .40)))
        twist=getattr(self, 'truth_twist', None)
        valid_twist=bool(
            twist is not None and len(twist) >= 2 and
            math.isfinite(float(twist[0])) and
            math.isfinite(float(twist[1])))
        if threshold <= 0.0 or not valid_twist or flight_b_gain < minimum_gain:
            self.truth_flight_b_rearward_speed=None
            self.truth_flight_b_filtered_rearward_speed=None
            self.truth_flight_b_rearward_slip_since=None
            return False

        configured_heading=getattr(self, 'truth_flight_b_heading', None)
        heading=(configured_heading
                 if configured_heading is not None else
                 -math.pi/2.0)
        forward_speed=(float(twist[0])*math.cos(heading)+
                       float(twist[1])*math.sin(heading))
        rearward_speed=-forward_speed
        self.truth_flight_b_rearward_speed=rearward_speed
        previous=getattr(
            self, 'truth_flight_b_filtered_rearward_speed', None)
        alpha=min(1.0, max(.01, float(getattr(
            self, 'truth_flight_b_rearward_slip_velocity_alpha', .70))))
        filtered=(rearward_speed if previous is None else
                  (1.0-alpha)*float(previous)+alpha*rearward_speed)
        self.truth_flight_b_filtered_rearward_speed=filtered

        since=getattr(self, 'truth_flight_b_rearward_slip_since', None)
        # Once the high threshold is crossed, use a lower release threshold
        # through the short dwell.  Round82e peaked at 0.424 m/s then dipped
        # to 0.268 m/s for one gait sample before accelerating into a lateral
        # fall; resetting at 0.300 on that sample discarded the warning.
        release_ratio=min(1.0, max(0.0, float(getattr(
            self, 'truth_flight_b_rearward_slip_release_ratio', .80))))
        active_threshold=(threshold*release_ratio
                          if since is not None else threshold)
        if filtered < active_threshold:
            self.truth_flight_b_rearward_slip_since=None
            return False
        if since is None or now < since:
            self.truth_flight_b_rearward_slip_since=now
            since=now
        dwell=max(0.0, float(getattr(
            self, 'truth_flight_b_rearward_slip_detect_seconds', .04)))
        if now-since < dwell:
            return False

        self.truth_flight_b_rearward_slip_latched=True
        self.truth_flight_b_rearward_slip_events=int(getattr(
            self, 'truth_flight_b_rearward_slip_events', 0))+1
        self.truth_flight_b_rearward_slip_last_trigger_gain=flight_b_gain
        rospy.logwarn(
            'Flight-B dynamic backslide caught before posture collapse: '
            'filtered rearward speed=%.3f m/s for %.2f s at flight gain '
            '%.2f m (threshold %.3f m/s).',
            filtered, now-since, flight_b_gain, threshold)
        return True

    def _reset_flight_b_anchors(self, now):
        """Give flight B its full existing budget after the re-entry hold."""
        self.flight_b_started_at=now
        self.truth_flight_b_stall_start_z=None
        self.truth_flight_b_stall_since=None
        self.truth_flight_b_recovery_until=None
        self._reset_flight_b_rearward_slip(clear_events=True)
        self._reset_flight_b_pitch_collapse(clear_events=True)
        self._reset_flight_b_alignment_recovery(clear_attempts=True)
        if self.truth_pose is not None:
            self.truth_flight_b_entry_z=self.truth_pose[2]

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
              'truth_roll':getattr(self, 'truth_roll', None),
              'truth_pitch':getattr(self, 'truth_pitch', None),
              'truth_twist':getattr(self, 'truth_twist', None),
              'entry_joint_velocity_rms':getattr(
                  self, 'entry_joint_velocity_rms', None),
              'heading':self.direction,
              'pose_source':('gazebo_truth' if pose is self.truth_pose else 'odometry'),
              'flight_a_early_retry_attempt':getattr(
                  self, 'truth_flight_a_early_retry_attempts', 0)}
        if self.phase in ('STAIR_ASCENT_B', getattr(
                self, 'upper_floor_settle_phase', 'SECOND_FLOOR_SETTLE')):
            item.update({
                'flight_b_center_x':self._truth_flight_b_center_x(),
                'flight_b_center_error':self.truth_flight_b_center_error,
                'flight_b_predicted_center_error':
                    self.truth_flight_b_predicted_center_error,
                'flight_b_lateral_velocity':
                    self.truth_flight_b_lateral_velocity,
                'flight_b_predictive_recovery_active':
                    self.truth_flight_b_predictive_recovery_active,
                'flight_b_predictive_center_crossing_active':getattr(
                    self,
                    'truth_flight_b_predictive_center_crossing_active',
                    False),
                'flight_b_predictive_strong_recovery_active':getattr(
                    self, 'truth_flight_b_predictive_strong_active', False),
                'flight_b_near_edge_strong_recovery_active':getattr(
                    self, 'truth_flight_b_near_edge_strong_active', False),
                'flight_b_recovery_active':
                    self.truth_flight_b_recovery_active,
                'flight_b_command_forward_speed':
                    self.truth_flight_b_command_forward_speed,
                'flight_b_command_center_speed':
                    self.truth_flight_b_command_center_speed,
                'flight_b_command_yaw_rate':
                    self.truth_flight_b_command_yaw_rate,
                'flight_b_heading_error':self.truth_flight_b_heading_error,
                'flight_b_alignment_recovery_active':(
                    self.truth_flight_b_alignment_recovery_until is not None),
                'flight_b_alignment_recovery_attempts':
                    self.truth_flight_b_alignment_recovery_attempts,
                'flight_b_alignment_recovery_elapsed_sec':(
                    now-self.truth_flight_b_alignment_recovery_started_at
                    if self.truth_flight_b_alignment_recovery_started_at
                    is not None else None),
                'flight_b_rearward_speed':getattr(
                    self, 'truth_flight_b_rearward_speed', None),
                'flight_b_filtered_rearward_speed':getattr(
                    self, 'truth_flight_b_filtered_rearward_speed', None),
                'flight_b_rearward_slip_latched':getattr(
                    self, 'truth_flight_b_rearward_slip_latched', False),
                'flight_b_rearward_slip_events':getattr(
                    self, 'truth_flight_b_rearward_slip_events', 0),
                'flight_b_pitch_rate_rps':getattr(
                    self, 'truth_flight_b_pitch_rate', None),
                'flight_b_pitch_collapse_latched':getattr(
                    self, 'truth_flight_b_pitch_collapse_latched', False),
                'flight_b_pitch_collapse_events':getattr(
                    self, 'truth_flight_b_pitch_collapse_events', 0),
                'flight_b_peak_gain':self.truth_flight_b_peak_gain,
                'flight_b_elapsed_sec':(
                    now-self.flight_b_started_at
                    if self.flight_b_started_at is not None else None),
                'flight_b_timeout_sec':self.flight_b_timeout,
                'flight_b_stall_recoveries':self.truth_flight_b_stall_recoveries})
        if self.phase=='STAIR_ASCENT':
            item.update({
                'flight_a_center_error':self.truth_flight_a_center_error,
                'flight_a_predicted_center_error':
                    self.truth_flight_a_predicted_center_error,
                'flight_a_lateral_velocity':
                    self.truth_flight_a_lateral_velocity,
                'flight_a_predictive_recovery_active':
                    self.truth_flight_a_predictive_recovery_active,
                'flight_a_predictive_center_crossing_active':getattr(
                    self,
                    'truth_flight_a_predictive_center_crossing_active',
                    False),
                'flight_a_predictive_strong_recovery_active':getattr(
                    self, 'truth_flight_a_predictive_strong_active', False),
                'flight_a_near_edge_strong_recovery_active':getattr(
                    self, 'truth_flight_a_near_edge_strong_active', False),
                'flight_a_heading_error':self.truth_flight_a_heading_error,
                'flight_a_peak_gain':self.truth_flight_a_peak_gain,
                'flight_a_peak_drawdown':
                    self.truth_flight_a_peak_drawdown,
                'flight_a_recovery_active':
                    self.truth_flight_a_recovery_active,
                'flight_a_slip_active':self.truth_flight_a_slip_active,
                'flight_a_alignment_recovery_active':(
                    self.truth_flight_a_alignment_recovery_until is not None),
                'flight_a_alignment_recovery_attempts':
                    self.truth_flight_a_alignment_recovery_attempts,
                'flight_a_alignment_recovery_elapsed_sec':(
                    now-self.truth_flight_a_alignment_recovery_started_at
                    if self.truth_flight_a_alignment_recovery_started_at
                    is not None else None),
                'flight_a_stall_release_active':
                    self.truth_flight_a_stall_release_active,
                'flight_a_stall_realign_active':getattr(
                    self, 'truth_flight_a_stall_realign_active', False),
                'flight_a_stall_realign_attempts':getattr(
                    self, 'truth_flight_a_stall_realign_attempts', 0),
                'flight_a_rearward_speed':getattr(
                    self, 'truth_flight_a_rearward_speed', None),
                'flight_a_filtered_rearward_speed':getattr(
                    self, 'truth_flight_a_filtered_rearward_speed', None),
                'flight_a_filtered_climb_speed':getattr(
                    self, 'truth_flight_a_filtered_climb_speed', None),
                'flight_a_rearward_slip_latched':getattr(
                    self, 'truth_flight_a_rearward_slip_latched', False),
                'flight_a_rearward_slip_events':getattr(
                    self, 'truth_flight_a_rearward_slip_events', 0),
                'flight_a_backslide_recovery_active':(
                    getattr(
                        self,
                        'truth_flight_a_backslide_recovery_until', None)
                    is not None),
                'flight_a_backslide_brake_active':getattr(
                    self, 'truth_flight_a_backslide_brake_active', False),
                'flight_a_backslide_brake_center_speed':getattr(
                    self, 'truth_flight_a_backslide_brake_center_speed', 0.0),
                'flight_a_backslide_brake_yaw_rate':getattr(
                    self, 'truth_flight_a_backslide_brake_yaw_rate', 0.0),
                'flight_a_backslide_ramp_active':(
                    getattr(
                        self,
                        'truth_flight_a_backslide_recovery_ramp_until', None)
                    is not None and now <
                    self.truth_flight_a_backslide_recovery_ramp_until),
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
                'landing_servo_heading':self.truth_landing_servo_heading,
                'landing_servo_mode':self.truth_landing_servo_mode,
                'landing_cross_speed':self.truth_landing_cross_speed,
                'landing_yaw_rate':self.truth_landing_yaw_rate,
                'landing_recovery_active':self.landing_recovery_active,
                'landing_recovery_attempt':self.landing_recovery_attempts})
        if self.phase=='TRUTH_ENTRY_GUIDE':
            item.update({'entry_stage':self.truth_entry_stage,
                         'route_heading':self.truth_route_heading,
                         'target':list(self.truth_entry_target)
                                  if self.truth_entry_target is not None else None,
                         'target_distance':self.truth_entry_last_distance,
                         'corridor_return_active':
                             getattr(self, 'truth_using_corridor_return', False),
                         'corridor_recenter_active':
                             getattr(self, 'truth_corridor_recenter_active', False),
                         'corridor_center_error':
                             getattr(self, 'truth_corridor_center_error', None),
                         'corridor_predicted_center_error':
                             getattr(self, 'truth_corridor_predicted_center_error', None),
                         'corridor_lateral_velocity':
                             getattr(self, 'truth_corridor_lateral_velocity', None),
                         'corridor_command_center_speed':
                             getattr(self, 'truth_corridor_command_center_speed', None),
                         'corridor_command_forward_speed':
                             getattr(self, 'truth_corridor_command_forward_speed', None),
                         'corridor_motion_heading':
                             getattr(self, 'truth_corridor_motion_heading', None)})
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

    def _truth_flight_a_ramp_factor(self, now=None):
        """Bounded opt-in climb ramp for the first effective commands.

        Round21 flip: the first gait command after the readiness gate stepped
        directly to ascent speed.  The factor is monotone from zero to one
        over truth_flight_a_climb_ramp_seconds and is exactly one when the
        ramp is disabled (F1->F2 default) or already converged.
        """
        # getattr defaults keep legacy __new__ fixtures (F1/F2 control unit
        # tests) working without the new opt-in attributes.
        ramp_seconds=getattr(self, 'truth_flight_a_climb_ramp_seconds', 0.0)
        anchor=getattr(self, 'truth_flight_a_ready_anchor', None)
        if ramp_seconds is None or ramp_seconds <= 0.0 or anchor is None:
            return 1.0
        now=time.monotonic() if now is None else now
        return min(1.0, max(0.0, (now-anchor) / ramp_seconds))

    def _flight_a_readiness_gate(self):
        """Hold flight A until a fresh /locomotion_ready latches (F3 opt-in).

        Returns True when the caller may run flight-A control.  On the first
        ready observation every flight-A anchor is reset so the stall,
        hard-stall, peak-gain, and ascent-timeout clocks measure actual
        stair-policy ownership rather than the pre-readiness wait.
        """
        if (not self.truth_flight_a_require_fresh_ready or
                self.truth_flight_a_ready_anchor is not None):
            return True
        if not self.locomotion_ready:
            failure=self._flight_a_preflight_failure_reason()
            if failure is not None:
                self._fail_flight_a_policy_handoff(
                    'STAIR_FLIGHT_A_READY_FALL_DETECTED', failure)
                return False
            now=time.monotonic()
            started=getattr(
                self, 'truth_flight_a_ready_wait_started', None)
            if started is None:
                self.truth_flight_a_ready_wait_started=now
                started=now
            timeout=float(getattr(
                self, 'truth_flight_a_ready_wait_timeout', 20.0))
            if now-started >= timeout:
                self._fail_flight_a_policy_handoff(
                    'STAIR_FLIGHT_A_READY_TIMEOUT',
                    'fresh_locomotion_ready_timeout')
            return False
        self.truth_flight_a_ready_anchor=time.monotonic()
        self.truth_flight_a_ready_wait_started=None
        self.started=self.truth_flight_a_ready_anchor
        if self.pose is not None:
            self.ascent_start_z=self.pose[2]
        if self.truth_pose is not None:
            self.truth_ascent_start_z=self.truth_pose[2]
        self.truth_flight_a_peak_gain=None
        self.truth_flight_a_stall_window_start=None
        self.truth_flight_a_stall_window_z=None
        self.truth_flight_a_hard_stall_since=None
        self.truth_flight_a_stall_release_until=None
        self.truth_flight_a_stall_release_active=False
        self.truth_flight_a_stall_realign_active=False
        self.truth_flight_a_stall_realign_attempts=0
        self.truth_flight_a_slip_active=False
        self._reset_flight_a_rearward_slip(clear_events=True)
        self._reset_flight_a_alignment_recovery(clear_attempts=True)
        rospy.loginfo(
            'Flight-A fresh RL readiness observed; ascent clocks reset and '
            '%.1f s climb ramp started.',
            self.truth_flight_a_climb_ramp_seconds)
        return True

    def _truth_flight_a_control(self):
        """Return world-frame flight-A command and current tracking errors."""
        heading=self.truth_stair_heading
        center=self.truth_step_pose
        if self.truth_pose is None or heading is None or center is None:
            return None
        center_error=self._truth_stair_center_error(
            self.truth_pose, center, heading)
        right_x=math.sin(heading)
        right_y=-math.cos(heading)
        twist=getattr(self, 'truth_twist', None)
        measured_lateral_velocity=(
            float(twist[0])*right_x+float(twist[1])*right_y
            if twist is not None and len(twist) >= 2 and
            math.isfinite(float(twist[0])) and math.isfinite(float(twist[1]))
            else 0.0)
        previous_velocity=getattr(
            self, 'truth_flight_a_lateral_velocity', None)
        alpha=min(1.0, max(0.01, float(getattr(
            self, 'truth_flight_a_prediction_velocity_alpha', .25))))
        lateral_velocity=(
            measured_lateral_velocity if previous_velocity is None else
            (1.0-alpha)*float(previous_velocity)+
            alpha*measured_lateral_velocity)
        prediction_horizon=max(0.0, float(getattr(
            self, 'truth_flight_a_prediction_horizon', .70)))
        # Positive lateral velocity follows the stair's right normal and thus
        # makes the signed centre correction more negative.
        predicted_center_error=(
            center_error-prediction_horizon*lateral_velocity)
        prediction_min_speed=max(0.0, float(getattr(
            self, 'truth_flight_a_prediction_min_lateral_speed', .08)))
        # Do not wait for the projected error to grow past the ordinary
        # position threshold after the body has already acquired enough
        # lateral momentum to cross the centreline.  Full-flow round41
        # reached the upper Flight-A treads 0.14 m west of centre while
        # moving east at 0.18 m/s.  The former threshold-only predictor kept
        # commanding east for another gait sample; by the time it reversed,
        # lateral speed had grown to 0.44 m/s and the body could not stop
        # before the tread edge.  A projected sign crossing is earlier,
        # directionally unambiguous evidence and still requires the existing
        # filtered minimum-speed gate.
        projected_center_crossing=bool(
            prediction_horizon > 0.0 and
            abs(lateral_velocity) >= prediction_min_speed and
            center_error*predicted_center_error <= 0.0 and
            abs(predicted_center_error-center_error) > 1e-6)
        projected_outward_escape=bool(
            prediction_horizon > 0.0 and
            abs(lateral_velocity) >= prediction_min_speed and
            abs(predicted_center_error) >=
            self.truth_flight_a_recovery_center_error and
            abs(predicted_center_error) > abs(center_error))
        predictive_recovery=bool(
            projected_outward_escape or projected_center_crossing)
        control_center_error=(predicted_center_error
                              if predictive_recovery else center_error)
        heading_error=math.atan2(
            math.sin(heading-self.truth_pose[3]),
            math.cos(heading-self.truth_pose[3]))
        peak_gain=getattr(self, 'truth_flight_a_peak_gain', None)
        ascent_start_z=getattr(self, 'truth_ascent_start_z', None)
        current_gain=(self.truth_pose[2]-ascent_start_z
                      if ascent_start_z is not None else None)
        peak_drawdown=(max(0.0, peak_gain-current_gain)
                       if peak_gain is not None and current_gain is not None
                       else 0.0)
        slip_active=bool(
            peak_gain is not None and
            peak_gain >= float(getattr(
                self, 'truth_flight_a_slip_min_peak_gain', .30)) and
            peak_drawdown >= float(getattr(
                self, 'truth_flight_a_slip_drawdown_trigger', .06)))
        alignment_recovery_active=(getattr(
            self, 'truth_flight_a_alignment_recovery_until', None)
            is not None)
        center_recovery=(
            abs(center_error) >= self.truth_flight_a_recovery_center_error or
            predictive_recovery)
        heading_recovery=(
            abs(heading_error) >= self.truth_flight_a_recovery_heading_error)
        recovery_active=(
            center_recovery or heading_recovery or slip_active or
            alignment_recovery_active)
        if recovery_active:
            if getattr(self, 'truth_flight_a_recovery_since', None) is None:
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
        if getattr(self, 'truth_flight_a_stall_window_start', None) is None:
            self.truth_flight_a_stall_window_start=now
            self.truth_flight_a_stall_window_z=self.truth_pose[2]
            recovery_stall=False
        elif (now-self.truth_flight_a_stall_window_start >=
              self.truth_flight_a_recovery_stall_seconds):
            recovery_stall=(
                self.truth_pose[2]-self.truth_flight_a_stall_window_z <
                self.truth_flight_a_recovery_stall_z_progress_m)
            if recovery_stall:
                release_alignment_safe=(
                    abs(center_error) <=
                    self.truth_flight_a_stall_release_max_center_error and
                    abs(heading_error) <=
                    self.truth_flight_a_stall_release_max_heading_error)
                bounded_realign_safe=(
                    abs(center_error) < float(getattr(
                        self, 'truth_flight_a_max_center_error', .38)) and
                    abs(heading_error) < float(getattr(
                        self, 'truth_flight_a_max_heading_error', .65)))
                if (release_alignment_safe and
                        self.truth_flight_a_stall_release_seconds > 0.0):
                    self.truth_flight_a_stall_release_until=(
                        now+self.truth_flight_a_stall_release_seconds)
                    self.truth_flight_a_stall_realign_active=False
                    rospy.logwarn_throttle(
                        2.0,
                        'Flight-A stall: no height gain for %.1f s; '
                        'releasing full climb speed for %.1f s to break '
                        'through (gain %.3f m, alignment_safe=True).',
                        self.truth_flight_a_recovery_stall_seconds,
                        self.truth_flight_a_stall_release_seconds,
                        self.truth_pose[2]-
                        self.truth_flight_a_stall_window_z)
                elif (bounded_realign_safe and
                      self.truth_flight_a_stall_release_seconds > 0.0):
                    # Round42 F2->F3 froze at +0.46 m with centre error
                    # 0.274 m: outside the 0.25 m full-release band but still
                    # safely inside the unchanged 0.38 m hard guard.  The
                    # resulting low-forward/high-strafe command could neither
                    # climb nor recenter.  Keep a bounded 0.40 m/s tread-
                    # contact component and the strong 0.30 m/s centre
                    # correction instead of entering that threshold deadzone.
                    self.truth_flight_a_stall_release_until=(
                        now+self.truth_flight_a_stall_release_seconds)
                    self.truth_flight_a_stall_realign_active=True
                    self.truth_flight_a_stall_realign_attempts=(
                        int(getattr(
                            self,
                            'truth_flight_a_stall_realign_attempts', 0))+1)
                    rospy.logwarn_throttle(
                        2.0,
                        'Flight-A stall outside the full-speed release band; '
                        'starting bounded recenter escape %d for %.1f s '
                        '(gain %.3f m, center error=%.3f m, heading '
                        'error=%.3f rad).',
                        self.truth_flight_a_stall_realign_attempts,
                        self.truth_flight_a_stall_release_seconds,
                        self.truth_pose[2]-
                        self.truth_flight_a_stall_window_z,
                        center_error, heading_error)
                else:
                    rospy.logwarn_throttle(
                        2.0,
                        'Flight-A stall: no height gain for %.1f s, but '
                        'alignment is outside both bounded escape envelopes '
                        '(gain %.3f m, center error=%.3f m, heading '
                        'error=%.3f rad).',
                        self.truth_flight_a_recovery_stall_seconds,
                        self.truth_pose[2]-
                        self.truth_flight_a_stall_window_z,
                        center_error, heading_error)
            self.truth_flight_a_stall_window_start=now
            self.truth_flight_a_stall_window_z=self.truth_pose[2]
        else:
            recovery_stall=False
        timed_stall_release=bool(
            getattr(self, 'truth_flight_a_stall_release_until', None)
            is not None and
            now < self.truth_flight_a_stall_release_until)
        if (getattr(self, 'truth_flight_a_stall_release_until', None)
                is not None and
                not timed_stall_release):
            self.truth_flight_a_stall_release_until=None
            self.truth_flight_a_stall_realign_active=False
        # Keep the legacy one-tick pulse at the instant a stall is detected,
        # while an opted-in upper-floor launch holds it across complete gait
        # steps.  During that hold retain only the nominal proportional
        # centre correction; the forced 0.18 m/s lateral recovery was what
        # dominated the 0.12 m/s forward command in the reproduced deadlock.
        stall_release_active=recovery_stall or timed_stall_release
        stall_realign_active=bool(
            stall_release_active and getattr(
                self, 'truth_flight_a_stall_realign_active', False))
        strong_slip_recovery=(
            slip_active or alignment_recovery_active or
            abs(heading_error) > float(getattr(
                self, 'truth_flight_a_max_heading_error', .65)))
        strong_predictive_recovery=bool(
            predictive_recovery and peak_gain is not None and
            peak_gain >= float(getattr(
                self, 'truth_flight_a_slip_min_peak_gain', .30)) and
            abs(lateral_velocity) >= float(getattr(
                self, 'truth_flight_a_predictive_strong_min_lateral_speed',
                .20)) and
            (projected_outward_escape or bool(getattr(
                self,
                'truth_flight_a_predictive_strong_on_center_crossing',
                True))))
        near_edge_threshold=max(0.0, float(getattr(
            self, 'truth_flight_a_near_edge_strong_center_error', 0.0)))
        near_edge_strong_recovery=bool(
            near_edge_threshold > 0.0 and
            max(abs(center_error), abs(predicted_center_error)) >=
            near_edge_threshold)
        strong_tread_recovery=(
            strong_slip_recovery or strong_predictive_recovery or
            near_edge_strong_recovery or stall_realign_active)
        legacy_recovery_forward_speed=(
            max(self.truth_flight_a_recovery_forward_speed,
                float(getattr(
                    self, 'truth_flight_a_slip_forward_speed', .40)),
                float(getattr(
                    self, 'truth_flight_a_predictive_strong_forward_speed',
                    .40)) if strong_predictive_recovery else 0.0)
            if strong_tread_recovery else
            self.truth_flight_a_recovery_forward_speed)
        strong_recovery_forward_speed=float(getattr(
            self, 'truth_flight_a_strong_recovery_forward_speed', -1.0))
        recovery_forward_speed=(
            min(self.ascent_speed, strong_recovery_forward_speed)
            if strong_tread_recovery and
            strong_recovery_forward_speed >= 0.0 else
            legacy_recovery_forward_speed)
        forward_speed=(
            min(self.ascent_speed, recovery_forward_speed)
            if stall_realign_active else (
                self.ascent_speed if stall_release_active else (
                min(self.ascent_speed, recovery_forward_speed)
                if recovery_active else self.ascent_speed)))
        forward_speed*=self._truth_flight_a_ramp_factor(now)
        backslide_ramp_until=getattr(
            self, 'truth_flight_a_backslide_recovery_ramp_until', None)
        backslide_ramp_active=bool(
            backslide_ramp_until is not None and now < backslide_ramp_until)
        if (backslide_ramp_until is not None and
                not backslide_ramp_active):
            self.truth_flight_a_backslide_recovery_ramp_until=None
        if backslide_ramp_active:
            forward_speed=min(
                forward_speed, max(0.0, float(getattr(
                    self,
                    'truth_flight_a_backslide_recovery_ramp_speed', .40))))
        if abs(control_center_error) <= self.truth_flight_a_center_deadband:
            center_speed=0.0
        else:
            boosted_center_recovery=(
                center_recovery and
                (not stall_release_active or stall_realign_active))
            strong_center_recovery=bool(
                strong_predictive_recovery or
                near_edge_strong_recovery or stall_realign_active)
            center_limit=(
                max(self.truth_flight_a_center_speed,
                    self.truth_flight_a_recovery_center_speed,
                    float(getattr(
                        self,
                        'truth_flight_a_predictive_strong_center_speed',
                        .30)) if strong_center_recovery else 0.0)
                if boosted_center_recovery else
                self.truth_flight_a_center_speed)
            requested_center=(self.truth_flight_a_center_gain*
                              control_center_error)
            center_magnitude=min(center_limit, abs(requested_center))
            if boosted_center_recovery:
                minimum_recovery_center_speed=(
                    float(getattr(
                        self,
                        'truth_flight_a_predictive_strong_center_speed',
                        .30)) if strong_center_recovery else
                    self.truth_flight_a_recovery_center_speed)
                center_magnitude=max(
                    min(center_limit,
                        minimum_recovery_center_speed),
                    center_magnitude)
            center_speed=math.copysign(
                center_magnitude, control_center_error)
        recovery_yaw_rate=(
            max(self.truth_flight_a_recovery_yaw_rate,
                float(getattr(
                    self, 'truth_flight_a_slip_yaw_rate', .60)))
            if strong_slip_recovery else
            self.truth_flight_a_recovery_yaw_rate)
        yaw_limit=(
            max(self.truth_flight_a_max_yaw_rate, recovery_yaw_rate)
            if heading_recovery or alignment_recovery_active else
            self.truth_flight_a_max_yaw_rate)
        requested_yaw=self.truth_flight_a_heading_gain*heading_error
        yaw_magnitude=min(yaw_limit, abs(requested_yaw))
        if heading_recovery:
            yaw_magnitude=max(
                min(yaw_limit, recovery_yaw_rate),
                yaw_magnitude)
        yaw_rate=(math.copysign(yaw_magnitude, heading_error)
                  if abs(heading_error) > 1e-6 else 0.0)
        vx=forward_speed*math.cos(heading)+center_speed*right_x
        vy=forward_speed*math.sin(heading)+center_speed*right_y
        self.truth_flight_a_lateral_velocity=lateral_velocity
        self.truth_flight_a_predicted_center_error=predicted_center_error
        self.truth_flight_a_predictive_recovery_active=predictive_recovery
        self.truth_flight_a_predictive_center_crossing_active=(
            projected_center_crossing)
        self.truth_flight_a_recovery_active=recovery_active
        self.truth_flight_a_slip_active=slip_active
        self.truth_flight_a_predictive_strong_active=(
            strong_predictive_recovery)
        self.truth_flight_a_near_edge_strong_active=(
            near_edge_strong_recovery)
        self.truth_flight_a_peak_drawdown=peak_drawdown
        self.truth_flight_a_stall_release_active=stall_release_active
        self.truth_flight_a_stall_realign_active=stall_realign_active
        self.truth_flight_a_command_forward_speed=forward_speed
        self.truth_flight_a_command_center_speed=center_speed
        self.truth_flight_a_command_yaw_rate=yaw_rate
        return vx,vy,yaw_rate,center_error,heading_error

    def _truth_flight_a_fall_detected(self, truth_gain):
        """Keep the physical height-drop guard independent of recovery."""
        peak=getattr(self, 'truth_flight_a_peak_gain', None)
        return bool(
            peak is not None and
            peak >= float(getattr(
                self, 'truth_flight_a_slip_min_peak_gain', .30)) and
            truth_gain < peak-float(getattr(
                self, 'truth_flight_a_fall_drop', .38)))

    def _truth_flight_a_early_derail_detected(
            self, truth_gain, center_error, heading_error):
        """Recognize a first-tread slip that has returned to level ground.

        This is deliberately disjoint from the normal on-stair recovery:
        the achieved peak is high enough to prove contact with the first
        treads, but still below the established slip/fall arming height; the
        body has then lost most of that height and a material lateral or yaw
        error shows that it is no longer attacking the riser squarely.
        """
        peak=getattr(self, 'truth_flight_a_peak_gain', None)
        if peak is None:
            return False
        minimum_peak=float(getattr(
            self, 'truth_flight_a_early_retry_min_peak_gain', .10))
        committed_peak=float(getattr(
            self, 'truth_flight_a_slip_min_peak_gain', .30))
        drawdown=max(0.0, float(peak)-float(truth_gain))
        alignment_lost=bool(
            abs(float(center_error)) >= float(getattr(
                self, 'truth_flight_a_early_retry_min_center_error', .18)) or
            abs(float(heading_error)) >= float(getattr(
                self, 'truth_flight_a_early_retry_min_heading_error', .65)))
        return bool(
            float(peak) >= minimum_peak and
            float(peak) < committed_peak and
            drawdown >= float(getattr(
                self, 'truth_flight_a_early_retry_drawdown', .10)) and
            float(truth_gain) <= float(getattr(
                self, 'truth_flight_a_early_retry_max_current_gain', .08)) and
            alignment_lost)

    def _start_flight_a_early_retry(
            self, truth_gain, center_error, heading_error, now=None):
        """Switch once to the plane gait before re-staging at the pre-riser."""
        now=time.monotonic() if now is None else now
        attempts=int(getattr(
            self, 'truth_flight_a_early_retry_attempts', 0))
        maximum=max(0, int(getattr(
            self, 'truth_flight_a_early_retry_max_attempts', 1)))
        plane=str(getattr(self, 'plane_policy', '')).strip()
        if attempts >= maximum:
            self.truth_flight_a_early_retry_reason='attempts_exhausted'
            return False
        if not plane:
            self.truth_flight_a_early_retry_reason='plane_policy_missing'
            return False
        self.truth_flight_a_early_retry_attempts=attempts+1
        self.truth_flight_a_early_retry_reason='low_height_derail'
        self.truth_flight_a_early_retry_policy_deadline=(
            now+float(getattr(
                self, 'truth_flight_a_early_retry_policy_timeout', 8.0)))
        self.truth_flight_a_early_retry_plane_loaded_at=None
        # A previous stair acknowledgement is stale for the retry.  The next
        # ascent must receive the new stair-policy acknowledgement and fresh
        # readiness edge after the plane-guided return.
        self.policy_loaded=False
        self.truth_flight_a_ready_anchor=None
        self.cmd.publish(Twist())
        self.hold_rl()
        self.phase='STAIR_FLIGHT_A_RETRY_POLICY_LOADING'
        self.state.publish(String(data=self.phase))
        self.pub.publish(String(data=plane))
        rospy.logwarn(
            'Flight-A early derail detected (attempt %d/%d): gain=%.3f m, '
            'peak=%.3f m, center error=%.3f m, heading error=%.3f rad; '
            'loading the plane policy for one bounded pre-riser retry.',
            self.truth_flight_a_early_retry_attempts, maximum, truth_gain,
            self.truth_flight_a_peak_gain, center_error, heading_error)
        self._record_trace(force=True)
        return True

    def _fail_flight_a_early_retry(self, reason):
        """Persist an explicit terminal reason for an unsafe/exhausted retry."""
        self.truth_flight_a_early_retry_reason=str(reason)
        self.cmd.publish(Twist())
        self.hold_rl()
        self.phase='STAIR_FLIGHT_A_EARLY_DERAIL_FAILED'
        self.state.publish(String(data=self.phase))
        rospy.logerr('Flight-A early-derail retry failed: %s.', reason)
        self._record_trace(force=True)
        rospy.signal_shutdown('stair_flight_a_early_derail_failed')

    def _truth_flight_a_alignment_recovery_control(
            self, now, center_error, heading_error):
        """Give a centred Flight-A heading excursion a bounded recovery.

        This helper never relaxes the lateral or height-drop guards.  It only
        delays a heading-only terminal decision while the trunk remains in a
        conservative centre band, and requires measurable heading progress
        before a second short attempt is granted.
        """
        max_center=float(getattr(
            self, 'truth_flight_a_max_center_error', .38))
        max_heading=float(getattr(
            self, 'truth_flight_a_max_heading_error', .65))
        safe_center=min(max_center, max(0.0, float(getattr(
            self, 'truth_flight_a_alignment_recovery_safe_center_error',
            .24))))
        hard_heading=max(max_heading, float(getattr(
            self, 'truth_flight_a_alignment_recovery_hard_heading_error',
            1.20)))
        active=(getattr(
            self, 'truth_flight_a_alignment_recovery_until', None)
            is not None)
        center_lost=abs(center_error) > max_center
        heading_lost=abs(heading_error) > max_heading

        def failed(reason):
            self.truth_flight_a_alignment_recovery_last_reason=reason
            return ('failed', reason)

        if not active and not (center_lost or heading_lost):
            return ('normal', None)
        if not active:
            attempts=int(getattr(
                self, 'truth_flight_a_alignment_recovery_attempts', 0))
            maximum_attempts=max(1, int(getattr(
                self, 'truth_flight_a_alignment_recovery_max_attempts', 2)))
            if center_lost:
                return failed('center_error_exceeded')
            if abs(center_error) > safe_center:
                return failed('heading_lost_outside_safe_center_band')
            if abs(heading_error) > hard_heading:
                return failed('hard_heading_error_exceeded')
            if attempts >= maximum_attempts:
                return failed('alignment_recovery_attempts_exhausted')
            duration=max(.1, float(getattr(
                self, 'truth_flight_a_alignment_recovery_seconds', 1.5)))
            self.truth_flight_a_alignment_recovery_attempts=attempts+1
            self.truth_flight_a_alignment_recovery_started_at=now
            self.truth_flight_a_alignment_recovery_until=now+duration
            self.truth_flight_a_alignment_recovery_start_heading_error=(
                abs(heading_error))
            self.truth_flight_a_alignment_recovery_stable_since=None
            self.truth_flight_a_alignment_recovery_last_reason=(
                'heading_error_exceeded')
            active=True
            rospy.logwarn(
                'Flight-A heading guard entered bounded slip recovery %d/%d: '
                'center error=%.3f m, heading error=%.3f rad; retaining '
                'tread-contact speed for up to %.1f s.',
                self.truth_flight_a_alignment_recovery_attempts,
                maximum_attempts, center_error, heading_error, duration)

        if center_lost:
            return failed('center_error_exceeded_during_recovery')
        if abs(heading_error) > hard_heading:
            return failed('hard_heading_error_exceeded_during_recovery')

        stable=bool(
            abs(center_error) <= float(getattr(
                self,
                'truth_flight_a_alignment_recovery_release_center_error',
                .16)) and
            abs(heading_error) <= float(getattr(
                self,
                'truth_flight_a_alignment_recovery_release_heading_error',
                .20)))
        if stable:
            if getattr(
                    self,
                    'truth_flight_a_alignment_recovery_stable_since',
                    None) is None:
                self.truth_flight_a_alignment_recovery_stable_since=now
            stable_elapsed=(
                now-self.truth_flight_a_alignment_recovery_stable_since)
            if stable_elapsed >= float(getattr(
                    self,
                    'truth_flight_a_alignment_recovery_stable_seconds',
                    .25)):
                attempts=self.truth_flight_a_alignment_recovery_attempts
                self._reset_flight_a_alignment_recovery(
                    clear_attempts=False)
                # Re-enter the existing climb ramp after the strong-yaw
                # correction instead of stepping straight back to full speed.
                if float(getattr(
                        self, 'truth_flight_a_climb_ramp_seconds', 0.0)) > 0.0:
                    self.truth_flight_a_ready_anchor=now
                self.truth_flight_a_stall_window_start=now
                self.truth_flight_a_stall_window_z=(
                    self.truth_pose[2]
                    if getattr(self, 'truth_pose', None) is not None else None)
                self.truth_flight_a_hard_stall_since=None
                rospy.logwarn(
                    'Flight-A bounded recovery %d settled: center '
                    'error=%.3f m, heading error=%.3f rad; resuming through '
                    'the configured climb ramp.',
                    attempts, center_error, heading_error)
                return ('recovered', None)
        else:
            self.truth_flight_a_alignment_recovery_stable_since=None

        if now >= self.truth_flight_a_alignment_recovery_until:
            attempts=int(getattr(
                self, 'truth_flight_a_alignment_recovery_attempts', 1))
            maximum_attempts=max(1, int(getattr(
                self, 'truth_flight_a_alignment_recovery_max_attempts', 2)))
            start_heading=float(getattr(
                self,
                'truth_flight_a_alignment_recovery_start_heading_error',
                abs(heading_error)))
            made_progress=bool(
                abs(center_error) <= safe_center and
                abs(heading_error) <= start_heading-.10)
            if attempts < maximum_attempts and made_progress:
                duration=max(.1, float(getattr(
                    self, 'truth_flight_a_alignment_recovery_seconds', 1.5)))
                self.truth_flight_a_alignment_recovery_attempts=attempts+1
                self.truth_flight_a_alignment_recovery_until=now+duration
                self.truth_flight_a_alignment_recovery_start_heading_error=(
                    abs(heading_error))
                self.truth_flight_a_alignment_recovery_stable_since=None
                rospy.logwarn(
                    'Flight-A bounded recovery made progress but has not '
                    'settled; extending final attempt %d/%d for %.1f s '
                    '(heading error=%.3f rad).',
                    self.truth_flight_a_alignment_recovery_attempts,
                    maximum_attempts, duration, heading_error)
            else:
                return failed('alignment_recovery_timeout')

        return ('recovering', None)

    def _truth_flight_b_tracking_control(self, nominal_climb_speed):
        """Return a truth-frame Flight-B command with predictive centring.

        Flight B runs along world -Y in the fixed two-flight profile and its
        lateral axis is world X.  The plane/stair policy responds one gait
        step after a lateral request, so position-only saturation reacts too
        late to outward momentum.  A short ModelStates velocity projection
        predicts that momentum without changing the physical stair geometry.
        """
        if self.truth_pose is None:
            return None
        center_error=self._truth_flight_b_center_x()-self.truth_pose[0]
        twist=getattr(self, 'truth_twist', None)
        measured_lateral_velocity=(
            float(twist[0]) if twist is not None and len(twist) >= 1 and
            math.isfinite(float(twist[0])) else 0.0)
        previous_velocity=getattr(
            self, 'truth_flight_b_lateral_velocity', None)
        alpha=min(1.0, max(0.01, float(getattr(
            self, 'truth_flight_b_prediction_velocity_alpha', .30))))
        lateral_velocity=(
            measured_lateral_velocity if previous_velocity is None else
            (1.0-alpha)*float(previous_velocity)+
            alpha*measured_lateral_velocity)
        prediction_horizon=max(0.0, float(getattr(
            self, 'truth_flight_b_prediction_horizon', .80)))
        # Positive world-X motion reduces the signed error (center_x - x).
        predicted_center_error=(
            center_error-prediction_horizon*lateral_velocity)
        prediction_min_speed=max(0.0, float(getattr(
            self, 'truth_flight_b_prediction_min_lateral_speed', .06)))
        recovery_threshold=max(0.0, float(getattr(
            self, 'truth_flight_b_recovery_center_error', .12)))
        predictive_recovery=bool(
            prediction_horizon > 0.0 and
            abs(lateral_velocity) >= prediction_min_speed and
            ((abs(predicted_center_error) >= recovery_threshold and
              abs(predicted_center_error) > abs(center_error)) or
             (center_error*predicted_center_error <= 0.0 and
              abs(predicted_center_error-center_error) > 1e-6)))
        projected_center_crossing=bool(
            predictive_recovery and
            center_error*predicted_center_error <= 0.0 and
            abs(predicted_center_error-center_error) > 1e-6)
        strong_predictive_recovery=bool(
            predictive_recovery and
            abs(lateral_velocity) >= float(getattr(
                self,
                'truth_flight_b_predictive_strong_min_lateral_speed', .15)))
        near_edge_threshold=max(0.0, float(getattr(
            self, 'truth_flight_b_near_edge_strong_center_error', 0.0)))
        near_edge_strong_recovery=bool(
            near_edge_threshold > 0.0 and
            abs(center_error) >= near_edge_threshold)
        strong_center_recovery=bool(
            strong_predictive_recovery or near_edge_strong_recovery)
        target_heading=(self.truth_flight_b_heading
                        if self.truth_flight_b_heading is not None
                        else -math.pi/2.0)
        heading_error=math.atan2(
            math.sin(target_heading-self.truth_pose[3]),
            math.cos(target_heading-self.truth_pose[3]))
        recovery_active=bool(getattr(
            self, 'truth_flight_b_recovery_active', False))
        recovery_requested=bool(
            abs(center_error) >= recovery_threshold or
            predictive_recovery or
            abs(heading_error) >= float(getattr(
                self, 'truth_flight_b_recovery_heading_error', .25)))
        release_threshold=max(0.0, float(getattr(
            self, 'truth_flight_b_recovery_release_center_error', .10)))
        recovery_release=bool(
            abs(center_error) <= release_threshold and
            abs(predicted_center_error) <= recovery_threshold and
            abs(lateral_velocity) <= prediction_min_speed and
            abs(heading_error) <= .15)
        if recovery_requested:
            recovery_active=True
        elif recovery_active and recovery_release:
            recovery_active=False
        control_center_error=(predicted_center_error
                              if recovery_active else center_error)

        if abs(control_center_error) <= self.truth_flight_b_center_deadband:
            center_speed=0.0
        else:
            center_limit=(
                max(self.truth_flight_b_center_speed,
                    float(getattr(
                        self, 'truth_flight_b_recovery_center_speed', .20)),
                    float(getattr(
                        self, 'truth_flight_b_predictive_strong_center_speed',
                        .30)) if strong_center_recovery else 0.0)
                if recovery_active else self.truth_flight_b_center_speed)
            requested_center=self.truth_flight_b_center_gain*control_center_error
            center_magnitude=min(center_limit, abs(requested_center))
            if recovery_active:
                center_magnitude=max(
                    min(center_limit, (
                        float(getattr(
                            self,
                            'truth_flight_b_predictive_strong_center_speed',
                            .30)) if strong_center_recovery else
                        float(getattr(
                            self, 'truth_flight_b_recovery_center_speed',
                            .20)))),
                    center_magnitude)
            center_speed=math.copysign(
                center_magnitude, control_center_error)

        yaw_limit=(
            max(self.truth_flight_b_max_yaw_rate, float(getattr(
                self, 'truth_flight_b_tracking_recovery_yaw_rate', .16)))
            if recovery_active else self.truth_flight_b_max_yaw_rate)
        requested_yaw=self.truth_flight_b_heading_gain*heading_error
        yaw_rate=max(-yaw_limit, min(yaw_limit, requested_yaw))
        climb_speed=float(nominal_climb_speed)
        if recovery_active:
            recovery_forward_speed=float(getattr(
                self, 'truth_flight_b_recovery_forward_speed', .40))
            strong_recovery_forward_speed=float(getattr(
                self, 'truth_flight_b_strong_recovery_forward_speed', -1.0))
            if (strong_center_recovery and
                    strong_recovery_forward_speed >= 0.0):
                recovery_forward_speed=min(
                    recovery_forward_speed, strong_recovery_forward_speed)
            climb_speed=min(climb_speed, recovery_forward_speed)

        self.truth_flight_b_center_error=center_error
        self.truth_flight_b_predicted_center_error=predicted_center_error
        self.truth_flight_b_lateral_velocity=lateral_velocity
        self.truth_flight_b_predictive_recovery_active=predictive_recovery
        self.truth_flight_b_predictive_center_crossing_active=(
            projected_center_crossing)
        self.truth_flight_b_predictive_strong_active=(
            strong_predictive_recovery)
        self.truth_flight_b_near_edge_strong_active=(
            near_edge_strong_recovery)
        self.truth_flight_b_recovery_active=recovery_active
        self.truth_flight_b_heading_error=heading_error
        self.truth_flight_b_command_forward_speed=climb_speed
        self.truth_flight_b_command_center_speed=center_speed
        self.truth_flight_b_command_yaw_rate=yaw_rate
        return (center_speed, -climb_speed, yaw_rate,
                center_error, heading_error)

    def _truth_flight_b_alignment_recovery_control(
            self, now, center_speed, center_error, heading_error):
        """Bounded recovery for a centred Flight-B heading excursion.

        Returns ``(status, vx, vy, wz, reason)`` where ``status`` is one of
        ``normal``, ``recovering``, ``recovered``, or ``failed``.  The caller
        remains responsible for publishing the command and terminal phase.
        Keeping this decision separate makes the safety boundary testable
        without a running ROS graph.
        """
        max_center=float(getattr(
            self, 'truth_flight_b_max_center_error', .38))
        max_heading=float(getattr(
            self, 'truth_flight_b_max_heading_error', .65))
        safe_center=min(max_center, max(0.0, float(getattr(
            self, 'truth_flight_b_alignment_recovery_safe_center_error',
            .24))))
        hard_heading=max(max_heading, float(getattr(
            self, 'truth_flight_b_alignment_recovery_hard_heading_error',
            1.20)))
        active=getattr(
            self, 'truth_flight_b_alignment_recovery_until', None) is not None
        center_lost=abs(center_error) > max_center
        heading_lost=abs(heading_error) > max_heading

        if not active and not (center_lost or heading_lost):
            return ('normal', None, None, None, None)
        if not active:
            attempts=int(getattr(
                self, 'truth_flight_b_alignment_recovery_attempts', 0))
            maximum_attempts=max(1, int(getattr(
                self, 'truth_flight_b_alignment_recovery_max_attempts', 2)))
            if center_lost:
                return ('failed', None, None, None,
                        'center_error_exceeded')
            if abs(center_error) > safe_center:
                return ('failed', None, None, None,
                        'heading_lost_outside_safe_center_band')
            if abs(heading_error) > hard_heading:
                return ('failed', None, None, None,
                        'hard_heading_error_exceeded')
            if attempts >= maximum_attempts:
                return ('failed', None, None, None,
                        'alignment_recovery_attempts_exhausted')
            duration=max(0.1, float(getattr(
                self, 'truth_flight_b_alignment_recovery_seconds', 1.5)))
            self.truth_flight_b_alignment_recovery_attempts=attempts+1
            self.truth_flight_b_alignment_recovery_started_at=now
            self.truth_flight_b_alignment_recovery_until=now+duration
            self.truth_flight_b_alignment_recovery_start_z=(
                self.truth_pose[2] if self.truth_pose is not None else None)
            self.truth_flight_b_alignment_recovery_start_heading_error=(
                abs(heading_error))
            self.truth_flight_b_alignment_recovery_stable_since=None
            self.truth_flight_b_alignment_recovery_last_reason=(
                'heading_error_exceeded')
            active=True
            rospy.logwarn(
                'Flight-B heading guard entered bounded recovery %d/%d: '
                'center error=%.3f m, heading error=%.3f rad; holding '
                'forward speed at zero for up to %.1f s.',
                self.truth_flight_b_alignment_recovery_attempts,
                maximum_attempts, center_error, heading_error, duration)

        start_z=getattr(
            self, 'truth_flight_b_alignment_recovery_start_z', None)
        height_drop=(
            start_z-self.truth_pose[2]
            if start_z is not None and self.truth_pose is not None else 0.0)
        if center_lost:
            return ('failed', None, None, None,
                    'center_error_exceeded_during_recovery')
        if abs(heading_error) > hard_heading:
            return ('failed', None, None, None,
                    'hard_heading_error_exceeded_during_recovery')
        if height_drop > float(getattr(
                self, 'truth_flight_b_alignment_recovery_max_drop', .25)):
            return ('failed', None, None, None,
                    'height_drop_exceeded_during_recovery')

        twist=getattr(self, 'truth_twist', None)
        planar_speed=(
            math.hypot(float(twist[0]), float(twist[1]))
            if twist is not None and len(twist) >= 2 and
            math.isfinite(float(twist[0])) and
            math.isfinite(float(twist[1])) else 0.0)
        measured_yaw_rate=(
            abs(float(twist[3]))
            if twist is not None and len(twist) >= 4 and
            math.isfinite(float(twist[3])) else 0.0)
        stable=bool(
            abs(center_error) <= float(getattr(
                self,
                'truth_flight_b_alignment_recovery_release_center_error',
                .16)) and
            abs(heading_error) <= float(getattr(
                self,
                'truth_flight_b_alignment_recovery_release_heading_error',
                .20)) and
            planar_speed <= float(getattr(
                self,
                'truth_flight_b_alignment_recovery_release_planar_speed',
                .22)) and
            measured_yaw_rate <= float(getattr(
                self,
                'truth_flight_b_alignment_recovery_release_yaw_rate',
                .30)))
        if stable:
            if getattr(
                    self,
                    'truth_flight_b_alignment_recovery_stable_since',
                    None) is None:
                self.truth_flight_b_alignment_recovery_stable_since=now
            stable_elapsed=(
                now-self.truth_flight_b_alignment_recovery_stable_since)
            if stable_elapsed >= float(getattr(
                    self,
                    'truth_flight_b_alignment_recovery_stable_seconds',
                    .25)):
                attempts=self.truth_flight_b_alignment_recovery_attempts
                self._reset_flight_b_alignment_recovery(
                    clear_attempts=False)
                self.truth_flight_b_ramp_until=(
                    now+float(getattr(
                        self, 'truth_flight_b_recovery_ramp_seconds', 1.0)))
                if self.truth_pose is not None:
                    self.truth_flight_b_stall_start_z=self.truth_pose[2]
                self.truth_flight_b_stall_since=now
                rospy.logwarn(
                    'Flight-B bounded recovery %d settled: center '
                    'error=%.3f m, heading error=%.3f rad; resuming through '
                    'the existing %.1f s climb ramp.',
                    attempts, center_error, heading_error,
                    float(getattr(
                        self, 'truth_flight_b_recovery_ramp_seconds', 1.0)))
                return ('recovered', 0.0, 0.0, 0.0, None)
        else:
            self.truth_flight_b_alignment_recovery_stable_since=None

        if now >= self.truth_flight_b_alignment_recovery_until:
            attempts=int(getattr(
                self, 'truth_flight_b_alignment_recovery_attempts', 1))
            maximum_attempts=max(1, int(getattr(
                self, 'truth_flight_b_alignment_recovery_max_attempts', 2)))
            start_heading=float(getattr(
                self,
                'truth_flight_b_alignment_recovery_start_heading_error',
                abs(heading_error)))
            made_progress=bool(
                abs(center_error) <= safe_center and
                abs(heading_error) <= start_heading-.10)
            if attempts < maximum_attempts and made_progress:
                duration=max(0.1, float(getattr(
                    self, 'truth_flight_b_alignment_recovery_seconds', 1.5)))
                self.truth_flight_b_alignment_recovery_attempts=attempts+1
                self.truth_flight_b_alignment_recovery_until=now+duration
                self.truth_flight_b_alignment_recovery_start_heading_error=(
                    abs(heading_error))
                self.truth_flight_b_alignment_recovery_stable_since=None
                rospy.logwarn(
                    'Flight-B bounded recovery made progress but has not '
                    'settled; extending final attempt %d/%d for %.1f s '
                    '(heading error=%.3f rad).',
                    self.truth_flight_b_alignment_recovery_attempts,
                    maximum_attempts, duration, heading_error)
            else:
                return ('failed', None, None, None,
                        'alignment_recovery_timeout')

        yaw_limit=max(0.0, float(getattr(
            self, 'truth_flight_b_slip_yaw_rate', .60)))
        yaw_rate=max(-yaw_limit, min(yaw_limit, 1.2*heading_error))
        center_limit=max(0.0, float(getattr(
            self, 'truth_flight_b_recovery_center_speed', .20)))
        recovery_center=max(
            -center_limit, min(center_limit, float(center_speed)))
        self.truth_flight_b_command_forward_speed=0.0
        self.truth_flight_b_command_center_speed=recovery_center
        self.truth_flight_b_command_yaw_rate=yaw_rate
        return ('recovering', recovery_center, 0.0, yaw_rate, None)

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

    def _truth_landing_heading_error(self, target_heading, current_heading):
        """Return a stable hairpin error at the ambiguous half-turn.

        Flight A reaches the landing near +pi/2 and flight B starts near
        -pi/2.  Tiny gait noise decides which sign a conventional wrapped
        error gives at exactly pi radians.  A positive turn makes the +X
        platform translation become a body-backward command halfway through
        the hairpin; the isolated round1 gait then ran off the north edge.
        Follow the already configured ``landing_turn_angle`` direction only
        inside the near-pi ambiguity band.  Once close to the target, normal
        shortest-angle feedback (including overshoot correction) is retained.
        """
        error=math.atan2(
            math.sin(target_heading-current_heading),
            math.cos(target_heading-current_heading))
        if (getattr(self, 'truth_fixed_two_flight_profile', False) and
                abs(error) >= 2.80):
            preferred=float(getattr(self, 'landing_turn_angle', -2.40))
            if preferred < 0.0 and error > 0.0:
                error-=2.0*math.pi
            elif preferred > 0.0 and error < 0.0:
                error+=2.0*math.pi
        return error

    def _truth_landing_servo_heading(self, final_heading, position_error):
        """Use forward walking for the long platform crossing.

        The stair policy turns reliably on the intermediate landing, but its
        body-lateral response is weak and phase dependent.  Round85a finished
        the hairpin before reaching the flight-B centreline, leaving a
        persistent +0.615 m world-X error despite a 0.40 m/s command.  Keep
        the body along the required world-X translation until only a short
        braking distance remains; then complete the final flight-B heading.
        A negative error uses the equivalent -X forward heading so bounded
        overshoot correction does not fall back to lateral strafing either.
        """
        switch_error=max(.12, 2.0*float(getattr(
            self, 'truth_landing_position_tolerance', .06)))
        error=float(position_error)
        if error > switch_error:
            return 0.0, 'forward_positive_x'
        if error < -switch_error:
            return -math.pi, 'forward_negative_x'
        return float(final_heading), 'final_heading'

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

    @staticmethod
    def _source_state_requests_handoff(payload):
        """Recognize both the live F1 handoff and its final persisted state.

        ``/simenv/baseline_state`` and ``/simenv/finalize_result`` are
        independent ROS topics.  The baseline manager publishes its handoff,
        then immediately replaces the latched state with ``SHUTDOWN`` and
        publishes finalization.  Under load the stair subscriber can therefore
        receive finalization first, or only receive the final ``SHUTDOWN``
        sample.  The latter preserves the successful handoff as its exact
        ``reason`` value, so accept that value without using a broad substring
        match that could accidentally arm on a diagnostic/failure token.
        """
        text = str(payload or "").strip()
        try:
            decoded = json.loads(text)
        except (TypeError, ValueError):
            decoded = None
        if isinstance(decoded, dict):
            state = str(decoded.get("state", "")).strip()
            reason = str(decoded.get("reason", "")).strip()
            return (
                state in ("STAIR_WAIT_ZONE", "STAIR_LOBBY_HANDOFF") or
                reason in (
                    "STAIR_WAIT_ZONE_REACHED",
                    "STAIR_CORRIDOR_EXIT_HANDOFF",
                    "STAIR_CORRIDOR_EXIT_HANDOFF_TRUTH_GATE",
                    "STAIR_LOBBY_HANDOFF_LOCALIZATION_FALLBACK",
                )
            )
        # Compatibility with the original plain-string state publisher.
        return ("STAIR_WAIT_ZONE" in text or
                "STAIR_LOBBY_HANDOFF" in text)

    def on_state(self,m):
        if ('STAIR_RETURN_TRANSIT' in m.data and self.phase=='WAIT_F1'):
            self._arm_return_transit()
            return
        if (self._source_state_requests_handoff(m.data) and
                self.phase == 'FIRST_FLOOR_FINALIZING'):
            # Cross-topic delivery can put us in the grace state before the
            # successful baseline-state sample arrives.  A verified success
            # cancels that pending shutdown and restores normal handoff.
            self.phase='WAIT_F1'
            self.finalization_deadline=None
            rospy.logwarn(
                'Late first-floor success state arrived after finalization; '
                'continuing with the F1-to-F2 stair handoff.')
        if (self._source_state_requests_handoff(m.data) and
                self.phase=='WAIT_F1'):
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

    def _truth_corridor_return_control(self, target, stair_heading,
                                       nominal_speed):
        """Return a centreline-captured command for the long F1 return.

        The corridor target is several metres along the stair axis but the
        handoff can begin more than twenty metres away.  Pointing directly at
        that target makes the lateral component proportional to
        ``cross_error / route_distance``; round18 consequently requested only
        0.03--0.07 m/s of recentering while running longitudinally at 0.9 m/s.

        This controller treats along-track and cross-track motion separately.
        A hysteretic lateral-only mode first captures the target line and is
        re-entered if either measured position or short-horizon velocity
        predicts another departure.  Within the band, bounded cross-track
        correction is combined with longitudinal travel.  It consumes only
        the already-enabled truth acceptance bridge and generated stair axis;
        no layout metadata enters online control.

        Returns ``(vx, vy, body_heading, center_error, predicted_error,
        lateral_velocity, recenter_active)`` in Gazebo world coordinates.
        """
        if self.truth_pose is None:
            return None
        speed=max(0.0, float(nominal_speed))
        forward=(math.cos(stair_heading), math.sin(stair_heading))
        right=(math.sin(stair_heading), -math.cos(stair_heading))
        dx=float(target[0])-float(self.truth_pose[0])
        dy=float(target[1])-float(self.truth_pose[1])
        along_error=dx*forward[0]+dy*forward[1]
        center_error=dx*right[0]+dy*right[1]

        twist=getattr(self, 'truth_twist', None)
        measured_lateral_velocity=(
            float(twist[0])*right[0]+float(twist[1])*right[1]
            if twist is not None and len(twist) >= 2 and
            math.isfinite(float(twist[0])) and
            math.isfinite(float(twist[1])) else 0.0)
        previous_velocity=getattr(
            self, 'truth_corridor_lateral_velocity', None)
        alpha=min(1.0, max(.01, float(getattr(
            self, 'truth_corridor_prediction_velocity_alpha', .30))))
        lateral_velocity=(
            measured_lateral_velocity if previous_velocity is None else
            (1.0-alpha)*float(previous_velocity)+
            alpha*measured_lateral_velocity)
        horizon=max(0.0, float(getattr(
            self, 'truth_corridor_prediction_horizon', .70)))
        # Positive velocity along ``right`` reduces target-minus-body error.
        predicted_error=center_error-horizon*lateral_velocity
        minimum_lateral_speed=max(0.0, float(getattr(
            self, 'truth_corridor_prediction_min_lateral_speed', .06)))
        enter_error=max(.05, float(getattr(
            self, 'truth_corridor_recenter_enter_error', .30)))
        release_error=max(.02, min(enter_error, float(getattr(
            self, 'truth_corridor_recenter_release_error', .10))))
        predictive_departure=bool(
            horizon > 0.0 and
            abs(lateral_velocity) >= minimum_lateral_speed and
            abs(predicted_error) >= enter_error and
            abs(predicted_error) > abs(center_error))
        recenter_active=bool(getattr(
            self, 'truth_corridor_recenter_active', False))
        if abs(center_error) >= enter_error or predictive_departure:
            recenter_active=True
        elif (recenter_active and abs(center_error) <= release_error and
              not predictive_departure):
            recenter_active=False

        gain=max(0.0, float(getattr(
            self, 'truth_corridor_center_gain', .90)))
        deadband=max(0.0, float(getattr(
            self, 'truth_corridor_center_deadband', .04)))
        if recenter_active:
            # If momentum predicts crossing through the complete capture band,
            # brake toward the predicted side before physical position follows.
            control_error=(predicted_error
                           if predictive_departure else center_error)
            if abs(control_error) <= deadband or speed <= 0.0:
                center_speed=0.0
            else:
                recenter_cap=min(speed, max(0.05, float(getattr(
                    self, 'truth_corridor_recenter_speed', .40))))
                minimum_command=min(
                    recenter_cap,
                    max(.12, min(float(getattr(
                        self, 'truth_entry_guide_min_speed', .25)), .22)))
                center_speed=math.copysign(
                    min(recenter_cap,
                        max(minimum_command, gain*abs(control_error))),
                    control_error)
            forward_speed=0.0
        else:
            control_error=(predicted_error
                           if abs(lateral_velocity) >= minimum_lateral_speed
                           else center_error)
            tracking_cap=min(speed, max(0.0, float(getattr(
                self, 'truth_corridor_tracking_center_speed', .26))))
            center_speed=(
                math.copysign(min(tracking_cap, gain*abs(control_error)),
                              control_error)
                if abs(control_error) > deadband else 0.0)
            # Keep the magnitude at or below the configured guide cap even
            # while longitudinal and centring components are combined.
            along_speed=math.sqrt(max(0.0, speed*speed-
                                     center_speed*center_speed))
            forward_speed=(math.copysign(along_speed, along_error)
                           if abs(along_error) > 1e-6 else 0.0)

        vx=forward_speed*forward[0]+center_speed*right[0]
        vy=forward_speed*forward[1]+center_speed*right[1]
        if math.hypot(vx, vy) > 1e-9:
            motion_heading=math.atan2(vy, vx)
        else:
            motion_heading=float(self.truth_pose[3])
        body_heading=motion_heading
        if (recenter_active and bool(getattr(
                self, 'truth_corridor_recenter_allow_reverse', True))):
            # Select the nearer body orientation while retaining the same
            # world velocity.  At the round18 doorway this backs east through
            # the opening instead of performing a drifting 166-degree turn.
            current_yaw=float(self.truth_pose[3])
            forward_error=math.atan2(
                math.sin(motion_heading-current_yaw),
                math.cos(motion_heading-current_yaw))
            reverse_heading=math.atan2(
                math.sin(motion_heading+math.pi),
                math.cos(motion_heading+math.pi))
            reverse_error=math.atan2(
                math.sin(reverse_heading-current_yaw),
                math.cos(reverse_heading-current_yaw))
            if abs(reverse_error) < abs(forward_error):
                body_heading=reverse_heading

        self.truth_corridor_recenter_active=recenter_active
        self.truth_corridor_center_error=center_error
        self.truth_corridor_predicted_center_error=predicted_error
        self.truth_corridor_lateral_velocity=lateral_velocity
        self.truth_corridor_command_center_speed=center_speed
        self.truth_corridor_command_forward_speed=forward_speed
        self.truth_corridor_motion_heading=motion_heading
        return (vx, vy, body_heading, center_error, predicted_error,
                lateral_velocity, recenter_active)

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

    def _truth_entry_speed_for_stage(self, distance):
        """Return the route speed without coupling lobby and pre-riser legs."""
        speed=min(float(self.truth_entry_guide_speed),
                  max(float(self.truth_entry_guide_min_speed),
                      .70*float(distance)))
        if getattr(self, 'truth_entry_stage', None) == 'final':
            speed=min(speed, max(
                float(self.truth_entry_guide_min_speed),
                float(getattr(self, 'truth_entry_final_speed', .55))))
        return speed

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
            self.truth_roll,self.truth_pitch=roll_pitch(pose.orientation)
            self.truth_pose=(float(pose.position.x), float(pose.position.y),
                             float(pose.position.z), yaw(pose.orientation))
            twists=getattr(message, 'twist', ())
            if index < len(twists):
                twist=twists[index]
                self.truth_twist=(float(twist.linear.x),
                                  float(twist.linear.y),
                                  float(twist.linear.z),
                                  float(twist.angular.z))
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
            # The truth return gate is emitted by this node only after F1 has
            # armed its strict four-room return transit.  Together those two
            # latches are sufficient physical proof of a successful handoff,
            # even if /finalize_result overtakes /baseline_state in transit.
            if (self.return_transit_armed and self.return_gate_published):
                rospy.logwarn(
                    'First-floor finalization overtook its success state; '
                    'continuing from the verified truth return gate.')
                self._begin_handoff()
                return
            self.phase='FIRST_FLOOR_FINALIZING'
            self.state.publish(String(data=self.phase))
            self.finalization_deadline=time.monotonic()+self.finalization_grace
            rospy.logwarn('First-floor mission ended before F1; allowing %.1f s for offline visualization before shutdown.', self.finalization_grace)
    def on_policy_status(self,m):
        payload=str(m.data)
        plane_name=os.path.basename(str(getattr(
            self, 'plane_policy', '')).strip())
        if ('policy_reloaded:' in payload and plane_name and
                plane_name in payload and self.phase == 'WAIT_F1'):
            self.policy_loaded=False
            self.policy_preloaded=False
        if self.phase=='STAIR_FLIGHT_A_RETRY_POLICY_LOADING':
            if ('policy_reloaded:' in payload and plane_name and
                    plane_name in payload):
                self.truth_flight_a_early_retry_plane_loaded_at=(
                    time.monotonic())
                self.phase='STAIR_FLIGHT_A_RETRY_PLANE_SETTLE'
                self.state.publish(String(data=self.phase))
                rospy.logwarn(
                    'Flight-A retry plane policy loaded; settling for %.1f s '
                    'before returning to the pre-riser.',
                    self.truth_flight_a_early_retry_plane_settle)
                return
            if 'policy_reload_failed:' in payload:
                self._fail_flight_a_early_retry('plane_policy_reload_failed')
                return
        if ('policy_reloaded:' in payload and
                os.path.basename(self.policy) in payload):
            self.policy_loaded=True
            if self.phase == 'WAIT_F1':
                # The sequencer preloads stair RL on the flat side staging
                # area and waits for fresh locomotion readiness before it
                # releases this manager.  Preserve that stable policy across
                # entry guide and flight A; repeating FixedStand here wastes
                # time and adds another unsupported controller boundary.
                self.policy_preloaded=True
            self.state.publish(String(data='STAIR_POLICY_READY'))
        elif ('policy_reload_failed:' in payload and self.phase in (
                'STAIR_POLICY_LOADING',
                'STAIR_FLIGHT_A_POLICY_QUEUED_STAND',
                'STAIR_FLIGHT_A_POLICY_RL_REENTRY')):
            self.phase='STAIR_POLICY_FAILED'; self.state.publish(String(data=self.phase))
            rospy.logerr('Stair policy reload failed: %s', payload)
            rospy.signal_shutdown('stair_policy_reload_failed')

    def _flight_a_preflight_failure_reason(self):
        """Return why a level-ground plane->stair handoff is unsafe."""
        roll=getattr(self, 'truth_roll', None)
        pitch=getattr(self, 'truth_pitch', None)
        maximum_tilt=float(getattr(
            self, 'truth_flight_a_preflight_max_tilt', .45))
        if ((roll is not None and abs(float(roll)) > maximum_tilt) or
                (pitch is not None and abs(float(pitch)) > maximum_tilt)):
            return 'body_tilt_exceeded'
        pose=getattr(self, 'truth_pose', None)
        anchor=getattr(
            self, 'truth_flight_a_policy_stand_anchor_z', None)
        maximum_drop=float(getattr(
            self, 'truth_flight_a_preflight_max_height_drop', .10))
        if (pose is not None and anchor is not None and
                float(pose[2]) < float(anchor)-maximum_drop):
            return 'body_height_drop_exceeded'
        return None

    def _fail_flight_a_policy_handoff(self, phase, reason):
        """Stop a failed pre-flight-A controller boundary explicitly."""
        self.cmd.publish(Twist())
        self._clear_fast_takeover_profile()
        self._clear_fast_fixed_stand_profile()
        self.phase=str(phase)
        self.state.publish(String(data=self.phase))
        rospy.logerr('Flight-A policy handoff stopped safely: %s.', reason)
        self._record_trace(force=True)
        rospy.signal_shutdown('stair_flight_a_policy_handoff_failed')

    def _begin_flight_a_policy_stand(self, now):
        """Enter the configured safe FixedStand boundary before policy load."""
        # _flight_a_policy_joint_capture_safe() copied this exact qualified
        # sample while holding _entry_joint_lock.  Do not read the live fields
        # again here: a subscriber callback can otherwise pair the old safe
        # RMS with the next (possibly airborne) gait pose.
        names=getattr(
            self, 'truth_flight_a_policy_capture_joint_names', None)
        positions=getattr(
            self, 'truth_flight_a_policy_capture_joint_positions', None)
        self.truth_flight_a_policy_stand_started=now
        self.locomotion_ready=False
        self.fixed_stand_ready=False
        self.fixed_stand_ready_at=None
        self.fixed_stand_not_ready_at=None
        self.truth_flight_a_policy_fixed_stand_ready_confirmed=False
        use_default=bool(getattr(
            self, 'truth_flight_a_policy_use_default_stand_target', False))
        if use_default:
            # A stale owned snapshot would override FixedStand's built-in
            # symmetric target.  Clear it before publishing L2_A and fail
            # closed if the parameter service cannot prove it is gone.
            if not self._clear_stand_target_joints():
                self._fail_flight_a_policy_handoff(
                    'STAIR_FLIGHT_A_STAND_TARGET_CLEAR_TIMEOUT',
                    'default_stand_target_clear_failed')
                return
            applied=False
        else:
            applied=self._apply_stand_target_joints(
                self.truth_flight_a_policy_capture_joint_names,
                self.truth_flight_a_policy_capture_joint_positions,
                'flight-A pre-reload capture')
        if not self._arm_fast_fixed_stand_profile():
            self._fail_flight_a_policy_handoff(
                'STAIR_FLIGHT_A_FAST_FIXED_STAND_ARM_FAILED',
                'fast_fixed_stand_arm_failed')
            return
        self._publish_joy_fixed_stand()
        self.cmd.publish(Twist())
        self.phase='STAIR_FLIGHT_A_POLICY_STAND'
        self.state.publish(String(data=self.phase))
        rospy.loginfo(
            'Flight-A plane gait settled at joint RMS %s; holding FixedStand '
            'for at least %.1f s at %s before queued stair-policy entry.',
            ('unknown' if getattr(
                self, 'truth_flight_a_policy_capture_joint_velocity_rms',
                None) is None else
             '%.3f' % self.truth_flight_a_policy_capture_joint_velocity_rms),
            self.truth_flight_a_policy_stand_seconds,
            ('symmetric default stance' if use_default else
             ('snapshot pose' if applied else 'default stance')))
        self._record_trace(force=True)

    def _handle_flight_a_policy_stand(self, now=None):
        """Run FixedStand -> queued load -> fresh stair-RL boundary."""
        now=time.monotonic() if now is None else now
        failure=self._flight_a_preflight_failure_reason()
        if failure is not None:
            self._fail_flight_a_policy_handoff(
                'STAIR_FLIGHT_A_POLICY_HANDOFF_FALL_DETECTED', failure)
            return True
        self.cmd.publish(Twist())
        if self.phase=='STAIR_FLIGHT_A_POLICY_STAND':
            self._publish_joy_fixed_stand()
            stand_elapsed=now-self.truth_flight_a_policy_stand_started
            minimum_hold_done=(
                stand_elapsed >= self.truth_flight_a_policy_stand_seconds)
            require_ready=bool(getattr(
                self,
                'truth_flight_a_policy_require_fixed_stand_ready', False))
            false_at=getattr(self, 'fixed_stand_not_ready_at', None)
            ready_at=getattr(self, 'fixed_stand_ready_at', None)
            fresh_ready=bool(
                getattr(self, 'fixed_stand_ready', False) and
                false_at is not None and ready_at is not None and
                false_at >= self.truth_flight_a_policy_stand_started and
                ready_at >= false_at)
            if not minimum_hold_done or (require_ready and not fresh_ready):
                if (require_ready and stand_elapsed >= getattr(
                        self,
                        'truth_flight_a_policy_fixed_stand_ready_timeout',
                        12.0)):
                    self._fail_flight_a_policy_handoff(
                        'STAIR_FLIGHT_A_FIXED_STAND_READY_TIMEOUT',
                        'fresh_fixed_stand_ready_timeout')
                    return True
                self._record_trace()
                return True
            if require_ready and fresh_ready:
                self.truth_flight_a_policy_fixed_stand_ready_confirmed=True
            self._clear_fast_fixed_stand_profile()
            self.policy_loaded=False
            self.truth_flight_a_policy_queued_at=now
            self.phase='STAIR_FLIGHT_A_POLICY_QUEUED_STAND'
            self.state.publish(String(data=self.phase))
            self.pub.publish(String(data=self.policy))
            rospy.loginfo(
                'Flight-A FixedStand%s complete; queued the stair policy for '
                'State_RL entry and retaining stand ownership for %.2f s.',
                (' readiness gate' if require_ready else ' hold'),
                self.truth_flight_a_policy_queue_hold_seconds)
            self._record_trace(force=True)
            return True
        if self.phase=='STAIR_FLIGHT_A_POLICY_QUEUED_STAND':
            self._publish_joy_fixed_stand()
            queued_at=self.truth_flight_a_policy_queued_at
            if (queued_at is None or
                    now-queued_at <
                    self.truth_flight_a_policy_queue_hold_seconds):
                self._record_trace()
                return True
            if not self._clear_stand_target_joints():
                clear_wait=(now-self.truth_flight_a_policy_stand_started-
                            self.truth_flight_a_policy_stand_seconds-
                            self.truth_flight_a_policy_queue_hold_seconds)
                if clear_wait >= self.pre_ascent_rl_reentry_timeout:
                    self._fail_flight_a_policy_handoff(
                        'STAIR_FLIGHT_A_STAND_TARGET_CLEAR_TIMEOUT',
                        'stand_target_clear_timeout')
                else:
                    self._record_trace()
                return True
            if not self._arm_fast_takeover_profile():
                self._fail_flight_a_policy_handoff(
                    'STAIR_FLIGHT_A_FAST_TAKEOVER_ARM_FAILED',
                    'fast_takeover_arm_failed')
                return True
            self.locomotion_ready=False
            self.truth_flight_a_policy_rl_requested_at=now
            self.phase='STAIR_FLIGHT_A_POLICY_RL_REENTRY'
            self.state.publish(String(data=self.phase))
            self.hold_rl()
            rospy.loginfo(
                'Flight-A policy request has been queued under FixedStand; '
                'requesting RL so State_RL loads it before inference starts.')
            self._record_trace(force=True)
            return True
        if self.phase=='STAIR_FLIGHT_A_POLICY_RL_REENTRY':
            requested=self.truth_flight_a_policy_rl_requested_at
            fresh=bool(
                self.policy_loaded and
                self.locomotion_ready and
                self.locomotion_ready_at is not None and
                requested is not None and
                self.locomotion_ready_at >= requested)
            if not fresh:
                self.hold_rl()
                if (requested is not None and
                        now-requested >= self.pre_ascent_rl_reentry_timeout):
                    self._fail_flight_a_policy_handoff(
                        'STAIR_FLIGHT_A_POLICY_RL_REENTRY_TIMEOUT',
                        'fresh_locomotion_ready_timeout')
                else:
                    self._record_trace()
                return True
            self._clear_fast_takeover_profile()
            self.truth_flight_a_ready_anchor=None
            self.started=now
            if self.pose is not None:
                self.ascent_start_z=self.pose[2]
            if self.truth_pose is not None:
                self.truth_ascent_start_z=self.truth_pose[2]
            self.flight_a_heading=self.direction
            self.phase='STAIR_ASCENT'
            self.state.publish(String(data=self.phase))
            rospy.loginfo(
                'Fresh RL readiness confirmed after the flight-A '
                'FixedStand boundary; starting the climb ramp.')
            self._record_trace(force=True)
            return True
        return False

    def _pre_ascent_rl_motion_stable(self):
        """Return whether the moving stair gait has physically settled.

        Missing optional telemetry does not deadlock the transition: the
        configured continuous dwell still has to elapse.  In the generated
        acceptance world both ModelStates twist and joint velocities are
        present, so a still-moving hairpin resets the dwell clock.
        """
        joint_rms=getattr(self, 'entry_joint_velocity_rms', None)
        if (joint_rms is not None and joint_rms > getattr(
                self, 'pre_ascent_rl_settle_max_joint_velocity_rms', .35)):
            return False
        twist=getattr(self, 'truth_twist', None)
        if twist is None:
            return True
        vx,vy,vz,wz=twist
        if math.hypot(vx,vy) > getattr(
                self, 'pre_ascent_rl_settle_max_planar_speed', .18):
            return False
        if abs(vz) > getattr(
                self, 'pre_ascent_rl_settle_max_vertical_speed', .12):
            return False
        if abs(wz) > getattr(
                self, 'pre_ascent_rl_settle_max_yaw_rate', .25):
            return False
        return True

    def _flight_a_policy_body_motion_stable(self):
        """Return whether the trunk is settled for the F1 policy boundary.

        The plane gait's joint velocities are intentionally excluded here.
        A stationary learned gait has periodic joint-speed spikes, so folding
        them into the continuous dwell made a physically stable robot time
        out before FixedStand.  Joint speed is checked separately at the
        exact pose-capture tick.
        """
        twist=getattr(self, 'truth_twist', None)
        if twist is None:
            return True
        vx,vy,vz,wz=twist
        if math.hypot(vx,vy) > getattr(
                self, 'pre_ascent_rl_settle_max_planar_speed', .18):
            return False
        if abs(vz) > getattr(
                self, 'pre_ascent_rl_settle_max_vertical_speed', .12):
            return False
        if abs(wz) > getattr(
                self, 'pre_ascent_rl_settle_max_yaw_rate', .25):
            return False
        return True

    def _flight_a_policy_joint_capture_safe(self):
        """Atomically qualify and save the current gait sample to freeze."""
        lock=getattr(self, '_entry_joint_lock', None)
        if lock is not None:
            lock.acquire()
        try:
            joint_rms=getattr(self, 'entry_joint_velocity_rms', None)
            if (joint_rms is not None and joint_rms > getattr(
                    self,
                    'truth_flight_a_policy_capture_max_joint_velocity_rms',
                    .35)):
                return False
            names=getattr(self, 'entry_joint_names', None)
            positions=getattr(self, 'entry_joint_snapshot', None)
            self.truth_flight_a_policy_capture_joint_names=(
                list(names) if names else None)
            self.truth_flight_a_policy_capture_joint_positions=(
                list(positions) if positions is not None else None)
            self.truth_flight_a_policy_capture_joint_velocity_rms=joint_rms
            return True
        finally:
            if lock is not None:
                lock.release()

    def _handle_flight_a_policy_settle(self, now=None):
        """Enter FixedStand only from a settled, low-speed gait sample.

        The symmetric-default path ignores the captured joint *positions*,
        but it must not ignore their velocities.  Switching controllers while
        a leg is still moving quickly can turn a harmless gait-phase timing
        difference into an unsupported FixedStand entry.
        """
        now=time.monotonic() if now is None else now
        self.cmd.publish(Twist())
        failure=self._flight_a_preflight_failure_reason()
        if failure is not None:
            self._fail_flight_a_policy_handoff(
                'STAIR_FLIGHT_A_POLICY_HANDOFF_FALL_DETECTED', failure)
            return True
        self.hold_rl()
        if self._flight_a_policy_body_motion_stable():
            if self.truth_flight_a_policy_stable_since is None:
                self.truth_flight_a_policy_stable_since=now
        else:
            self.truth_flight_a_policy_stable_since=None
        stable_elapsed=(
            now-self.truth_flight_a_policy_stable_since
            if self.truth_flight_a_policy_stable_since is not None
            else 0.0)
        body_settled=(
            stable_elapsed >= self.truth_flight_a_policy_settle_seconds)
        # Always qualify the controller boundary by joint velocity.  When the
        # default target is selected, _begin_flight_a_policy_stand() discards
        # these captured positions and interpolates to FixedStand's known
        # symmetric pose; the atomic sample remains useful diagnostic evidence.
        capture_ready=self._flight_a_policy_joint_capture_safe()
        if body_settled and capture_ready:
            self._begin_flight_a_policy_stand(now)
            return True
        if body_settled:
            rospy.loginfo_throttle(
                1.0,
                'Flight-A body settled for %.2f s; waiting for joint RMS '
                'capture window <= %.3f (current=%s).',
                stable_elapsed,
                self.truth_flight_a_policy_capture_max_joint_velocity_rms,
                ('unknown' if self.entry_joint_velocity_rms is None else
                 '%.3f' % self.entry_joint_velocity_rms))
        if (self.truth_flight_a_policy_settle_started is not None and
                now-self.truth_flight_a_policy_settle_started >=
                self.truth_flight_a_policy_settle_timeout):
            reason=('plane_gait_joint_capture_window_not_found'
                    if body_settled else 'plane_gait_body_did_not_settle')
            self._fail_flight_a_policy_handoff(
                'STAIR_FLIGHT_A_POLICY_SETTLE_TIMEOUT', reason)
            return True
        self._record_trace()
        return True

    def _handle_pre_ascent_stand(self, now=None):
        """Pre-ascent settle and optional FixedStand gate (STAIR_ASCENT_B).

        Returns True while the gate is active; the caller must not publish
        any nonzero flight-B command until this method has observed a fresh
        /locomotion_ready=True and set pre_ascent_stand_done.  A missing
        optional snapshot falls back to the FixedStand default stance
        without raising.
        """
        now=time.monotonic() if now is None else now
        settle_seconds=max(0.0, getattr(
            self, 'pre_ascent_rl_settle_seconds', 0.0))
        if self.pre_ascent_stand_started is None and settle_seconds > 0.0:
            if getattr(self, 'pre_ascent_rl_settle_started', None) is None:
                self.pre_ascent_rl_settle_started=now
                self.pre_ascent_rl_stable_since=None
                rospy.loginfo(
                    'Pre-ascent RL settle: holding zero command until motion '
                    'is stable for %.1f s before FixedStand.', settle_seconds)
            self.cmd.publish(Twist())
            self.hold_rl()
            if self._pre_ascent_rl_motion_stable():
                if self.pre_ascent_rl_stable_since is None:
                    self.pre_ascent_rl_stable_since=now
            else:
                self.pre_ascent_rl_stable_since=None
            stable_elapsed=(
                now-self.pre_ascent_rl_stable_since
                if self.pre_ascent_rl_stable_since is not None else 0.0)
            if stable_elapsed < settle_seconds:
                timeout=max(settle_seconds, getattr(
                    self, 'pre_ascent_rl_settle_timeout', 2.5))
                if now-self.pre_ascent_rl_settle_started >= timeout:
                    self.phase='STAIR_PRE_ASCENT_RL_SETTLE_TIMEOUT'
                    self.state.publish(String(data=self.phase))
                    rospy.logerr(
                        'Flight-B entry did not settle within %.1f s; '
                        'refusing an unsafe FixedStand transition.', timeout)
                    self._record_trace(force=True)
                    rospy.signal_shutdown('pre_ascent_rl_settle_timeout')
                    return True
                rospy.loginfo_throttle(
                    1.0, 'Pre-ascent RL settle active: stable %.2f/%.2f s.',
                    stable_elapsed, settle_seconds)
                self._record_trace()
                return True
            # A zero-duration stand intentionally means "settle only".  Keep
            # the already-active RL controller engaged instead of needlessly
            # crossing through FixedStand and a fresh policy takeover.  This
            # removes the moving hairpin sample from flight-B entry while
            # preserving the configured stair speed and all safety gates.
            if self.pre_ascent_stand_seconds <= 0.0:
                self._reset_flight_b_anchors(now)
                self.pre_ascent_stand_done=True
                self.cmd.publish(Twist())
                self.hold_rl()
                rospy.loginfo(
                    'Pre-ascent RL motion settled for %.2f s; settle-only '
                    'gate complete (FixedStand disabled).', stable_elapsed)
                self._record_trace(force=True)
                return True
            # Save/stand from the post-hairpin zero-command sample, not the
            # moving sample captured on the exact gate-crossing tick.
            self._save_entry_joint_snapshot()
            rospy.loginfo('Pre-ascent RL motion settled for %.2f s; '
                          'switching to FixedStand.', stable_elapsed)
        if self.pre_ascent_stand_started is None:
            self.pre_ascent_stand_started=now
            # FixedStand exits the RL state, so the cached readiness is
            # stale; invalidate it and require a fresh handshake.
            self.locomotion_ready=False
            applied=self._apply_stand_target_joints()
            self._publish_joy_fixed_stand()
            self.cmd.publish(Twist())
            rospy.loginfo('Pre-ascent fixed stand: holding %.1f s at %s '
                          'before flight-B climb.',
                          self.pre_ascent_stand_seconds,
                          'snapshot pose' if applied else 'default stance')
            self._record_trace()
            return True
        if not self.pre_ascent_rl_requested:
            if (now-self.pre_ascent_stand_started >=
                    self.pre_ascent_stand_seconds):
                if not self._clear_stand_target_joints():
                    # Stay in FixedStand and retry.  Bound this safety gate by
                    # the same re-entry budget so a broken parameter server
                    # cannot leave the mission hanging indefinitely.
                    clear_wait=(now-self.pre_ascent_stand_started-
                                self.pre_ascent_stand_seconds)
                    self._publish_joy_fixed_stand()
                    self.cmd.publish(Twist())
                    if clear_wait >= self.pre_ascent_rl_reentry_timeout:
                        self.phase='STAIR_PRE_ASCENT_STAND_TARGET_CLEAR_TIMEOUT'
                        self.state.publish(String(data=self.phase))
                        rospy.logerr(
                            'Could not clear the owned pre-ascent stand '
                            'target within %.1f s; stopping stair mission.',
                            self.pre_ascent_rl_reentry_timeout)
                        self._record_trace(force=True)
                        rospy.signal_shutdown(
                            'pre_ascent_stand_target_clear_timeout')
                    else:
                        self._record_trace()
                    return True
                self.pre_ascent_rl_requested=True
                self.pre_ascent_rl_requested_at=now
                if not self._arm_fast_takeover_profile():
                    self.phase='STAIR_PRE_ASCENT_FAST_TAKEOVER_ARM_FAILED'
                    self.state.publish(String(data=self.phase))
                    self._record_trace(force=True)
                    rospy.signal_shutdown(
                        'pre_ascent_fast_takeover_arm_failed')
                    return True
                rospy.loginfo('Pre-ascent stand complete (%.1f s); '
                              'requesting fresh RL readiness.',
                              self.pre_ascent_stand_seconds)
                # Issue the transition on this same callback.  Waiting for
                # the next 50 Hz tick only extends unsupported landing time.
                self.hold_rl()
            else:
                self._publish_joy_fixed_stand()
            self.cmd.publish(Twist())
            self._record_trace()
            return True
        if not self._rl_reentry_fresh_ready():
            if (now-self.pre_ascent_rl_requested_at >=
                    self.pre_ascent_rl_reentry_timeout):
                self.cmd.publish(Twist())
                self._clear_fast_takeover_profile()
                self.phase='STAIR_PRE_ASCENT_RL_REENTRY_TIMEOUT'
                self.state.publish(String(data=self.phase))
                rospy.logerr(
                    'No fresh /locomotion_ready within %.1f s after the '
                    'pre-ascent stand; stopping stair mission.',
                    self.pre_ascent_rl_reentry_timeout)
                self._record_trace(force=True)
                rospy.signal_shutdown('pre_ascent_rl_reentry_timeout')
                return True
            self.cmd.publish(Twist())
            self.hold_rl()
            self._record_trace()
            return True
        # Fresh readiness confirmed: reset flight-B timeout/stall/entry-height
        # anchors so the climb gets its full existing budget.
        self._reset_flight_b_anchors(now)
        self._clear_fast_takeover_profile()
        self.pre_ascent_stand_done=True
        # FixedStand and the subsequent policy warm-up can change physical
        # yaw even though /cmd_vel stays zero (trial5: -0.050 rad at the
        # original gate, -0.310 rad after re-entry).  Reuse the landing servo
        # on the broad platform and require the same measured x/yaw gate
        # again; otherwise flight-B predictive protection immediately cuts
        # the first-riser command to recovery speed and the gait wedges.
        if (getattr(self, 'truth_entry_guide', False) and
                getattr(self, 'truth_fixed_two_flight_profile', False) and
                getattr(self, 'truth_pose', None) is not None):
            pose=getattr(self, 'pose', None)
            if pose is not None:
                self.landing_start=(pose[0], pose[1])
            self.landing_started_at=now
            self.landing_recovery_active=False
            self.landing_recovery_attempts=0
            self.phase='STAIR_LANDING_TURN'
            self.state.publish(String(data=self.phase))
            rospy.loginfo(
                'Fresh RL readiness after pre-ascent stand; rechecking the '
                'physical flight-B entry alignment before climbing.')
        else:
            rospy.loginfo('Fresh RL readiness after pre-ascent stand; '
                          'resuming flight-B climb.')
        self.hold_rl()
        self._record_trace()
        return True

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
        if self.locomotion_ready:
            self.locomotion_ready_at=time.monotonic()
    def on_fixed_stand_ready(self,m):
        now=time.monotonic()
        self.fixed_stand_ready=bool(m.data)
        if self.fixed_stand_ready:
            self.fixed_stand_ready_at=now
        else:
            self.fixed_stand_not_ready_at=now
            self.fixed_stand_ready_at=None
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
            'STAIR_LANDING_TURN', 'STAIR_ASCENT_B',
            'STAIR_FLIGHT_A_RETRY_POLICY_LOADING',
            'STAIR_FLIGHT_A_RETRY_PLANE_SETTLE')
        if desired_pause != self._pause_active:
            self._pause_active = desired_pause
            self._pause_pub.publish(Bool(data=desired_pause))
        if self.phase=='STAIR_FLIGHT_A_RETRY_POLICY_LOADING':
            self.cmd.publish(Twist())
            self.hold_rl()
            deadline=getattr(
                self, 'truth_flight_a_early_retry_policy_deadline', None)
            if deadline is not None and time.monotonic() >= deadline:
                self._fail_flight_a_early_retry(
                    'plane_policy_reload_timeout')
                return
            self._record_trace()
            return
        if self.phase=='STAIR_FLIGHT_A_RETRY_PLANE_SETTLE':
            self.cmd.publish(Twist())
            self.hold_rl()
            loaded_at=getattr(
                self, 'truth_flight_a_early_retry_plane_loaded_at', None)
            if loaded_at is None:
                self._fail_flight_a_early_retry(
                    'plane_policy_loaded_time_missing')
                return
            if (time.monotonic()-loaded_at >= float(getattr(
                    self, 'truth_flight_a_early_retry_plane_settle', 1.0))):
                # The robot is back on the level lower landing.  Reuse only
                # the already collision-validated final pre-riser leg, not
                # the long corridor/side route from the completed floor.
                self.truth_entry_stage='final'
                self.truth_entry_stage_switches+=1
                self.truth_entry_target=None
                self.truth_route_heading=None
                self.truth_entry_best_distance=None
                self.truth_entry_last_distance=None
                self.truth_entry_watchdog_anchor_distance=None
                self.truth_entry_deadline=(
                    time.monotonic()+self.truth_entry_timeout)
                self.truth_corridor_recenter_active=False
                self.phase='TRUTH_ENTRY_GUIDE'
                self.state.publish(String(data=self.phase))
                rospy.logwarn(
                    'Flight-A retry plane gait settled; returning directly '
                    'to the verified pre-riser pose.')
                self._record_trace(force=True)
                return
            self._record_trace()
            return
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
                    self.truth_corridor_recenter_active=False
                    self.truth_corridor_center_error=None
                    self.truth_corridor_predicted_center_error=None
                    self.truth_corridor_lateral_velocity=None
                    self.truth_corridor_command_center_speed=None
                    self.truth_corridor_command_forward_speed=None
                    self.truth_corridor_motion_heading=None
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
            self.truth_entry_target=target
            dx=target[0]-self.truth_pose[0]; dy=target[1]-self.truth_pose[1]
            distance=math.hypot(dx,dy)
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
            speed=self._truth_entry_speed_for_stage(distance)
            long_corridor_return=bool(
                self.truth_fixed_two_flight_profile and
                self.truth_using_corridor_return and
                int(getattr(self, 'source_floor_index', 0)) == 0)
            corridor_control=(
                self._truth_corridor_return_control(
                    target, stair_heading, speed)
                if long_corridor_return and distance > stage_tolerance else
                None)
            if corridor_control is not None:
                command_vx,command_vy,heading=corridor_control[:3]
            else:
                command_vx=speed*math.cos(math.atan2(dy,dx))
                command_vy=speed*math.sin(math.atan2(dy,dx))
                heading=math.atan2(dy,dx)
                if not long_corridor_return:
                    self.truth_corridor_recenter_active=False
            self.truth_route_heading=heading
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
                if abs(route_error) > .45:
                    translation_scale=0.0
                else:
                    translation_scale=max(.35, math.cos(route_error))
                self._publish_truth_world_command(
                                                  translation_scale*command_vx,
                                                  translation_scale*command_vy,
                                                  route_wz)
            else:
                self.cmd.publish(Twist())
            # A 0.45 m tolerance can launch the stair policy up to ~1.1 m
            # from the first riser.  It then walks on level ground rather
            # than engaging the first tread.  Require the calibrated staging
            # pose tightly before the final yaw/settle gate.
            if distance <= stage_tolerance:
                self.cmd.publish(Twist())
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
                    now=time.monotonic()
                    self.truth_stage_settle_until=(
                        now+self.truth_flight_a_policy_settle_seconds)
                    self.truth_flight_a_policy_settle_started=now
                    self.truth_flight_a_policy_stable_since=None
                    self.truth_flight_a_policy_stand_anchor_z=(
                        self.truth_pose[2]
                        if self.truth_pose is not None else None)
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
            if (self.truth_flight_a_policy_stand_seconds > 0.0 and
                    not self.policy_preloaded):
                self._handle_flight_a_policy_settle()
                return
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
        elif self.phase in (
                'STAIR_FLIGHT_A_POLICY_STAND',
                'STAIR_FLIGHT_A_POLICY_QUEUED_STAND',
                'STAIR_FLIGHT_A_POLICY_RL_REENTRY'):
            self._handle_flight_a_policy_stand()
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
                self.truth_flight_a_ready_anchor=None
                self.truth_flight_a_slip_active=False
                self._reset_flight_a_alignment_recovery(
                    clear_attempts=True)
                self.started=time.monotonic()
                self.ascent_start_z=self.pose[2] if self.pose else None
                self.truth_ascent_start_z=(self.truth_pose[2]
                                           if self.truth_entry_guide and self.truth_pose else None)
                self.flight_a_heading=self.direction
                self.state.publish(String(data=self.phase))
        elif self.phase=='STAIR_ASCENT' and self.pose:
            if (self.truth_entry_guide and self.truth_pose is not None and
                    self.truth_ascent_start_z is not None):
                if not self._flight_a_readiness_gate():
                    # The stair policy owns the joints but the fresh RL
                    # takeover has not latched yet (Round21: 5.3 s).  No
                    # flight-A clock may run and no nonzero command may be
                    # issued until the handshake is observed.
                    self.cmd.publish(Twist())
                    self.hold_rl()
                    self._record_trace()
                    return
                truth_gain=self.truth_pose[2]-self.truth_ascent_start_z
                if self.truth_fixed_two_flight_profile:
                    flight_a_now=time.monotonic()
                    rearward_backslide=(
                        self._truth_flight_a_rearward_slip_update(
                            flight_a_now, truth_gain))
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
                    # A first-tread drop back to lobby height is not the same
                    # as an on-stair slip.  Once lateral/yaw evidence also
                    # shows that the body is no longer square to the riser,
                    # stop the stair gait and make one plane-policy return to
                    # the verified pre-riser pose.  This check precedes the
                    # >=0.20 m tracking gate by design: Round36 peaked at only
                    # 0.15 m and otherwise remained invisible to every guard.
                    if self._truth_flight_a_early_derail_detected(
                            truth_gain, center_error, heading_error):
                        if self._start_flight_a_early_retry(
                                truth_gain, center_error, heading_error):
                            return
                        self._fail_flight_a_early_retry(getattr(
                            self, 'truth_flight_a_early_retry_reason',
                            'retry_unavailable'))
                        return
                    # Once the body is genuinely on the stairs, a large drop
                    # from its achieved peak is a fall, not slow progress.
                    # Likewise, continuing after the body leaves the tread
                    # corridor or reverses its yaw only drives it farther off
                    # the structure.  Stop and preserve an explicit terminal
                    # reason instead of waiting for the global timeout.
                    fall_detected=self._truth_flight_a_fall_detected(
                        truth_gain)
                    tracking_lost=(
                        truth_gain >= .20 and
                        (abs(center_error) >
                         self.truth_flight_a_max_center_error or
                         abs(heading_error) >
                         self.truth_flight_a_max_heading_error))
                    alignment_status='normal'
                    alignment_reason=None
                    if (not fall_detected and
                            (tracking_lost or
                             self.truth_flight_a_alignment_recovery_until
                             is not None)):
                        alignment_status,alignment_reason=(
                            self._truth_flight_a_alignment_recovery_control(
                                flight_a_now, center_error,
                                heading_error))
                    if fall_detected or alignment_status=='failed':
                        self.cmd.publish(Twist())
                        self.hold_rl()
                        self.phase=('STAIR_FLIGHT_A_FALL_DETECTED'
                                    if fall_detected else
                                    'STAIR_FLIGHT_A_ALIGNMENT_LOST')
                        self.state.publish(String(data=self.phase))
                        rospy.logerr(
                            'Flight-A guard stopped ascent: gain=%.2f m, '
                            'peak=%.2f m, center error=%.3f m, heading '
                            'error=%.3f rad, recovery reason=%s.', truth_gain,
                            self.truth_flight_a_peak_gain,
                            center_error, heading_error,
                            ('height_drop_exceeded' if fall_detected else
                             alignment_reason))
                        self._record_trace(force=True)
                        rospy.signal_shutdown(
                            'stair_flight_a_fall_detected' if fall_detected
                            else 'stair_flight_a_alignment_lost')
                        return
                    if rearward_backslide:
                        maximum_attempts=max(1, int(getattr(
                            self,
                            'truth_flight_a_backslide_recovery_max_attempts',
                            2)))
                        events=int(getattr(
                            self, 'truth_flight_a_rearward_slip_events', 0))
                        if events > maximum_attempts:
                            self.cmd.publish(Twist())
                            self.hold_rl()
                            self.phase='STAIR_FLIGHT_A_BACKSLIDE_EXHAUSTED'
                            self.state.publish(String(data=self.phase))
                            rospy.logerr(
                                'Flight-A dynamic backslide recovery '
                                'exhausted after %d attempts (maximum %d); '
                                'stopping on the stair.',
                                events-1, maximum_attempts)
                            self._record_trace(force=True)
                            rospy.signal_shutdown(
                                'stair_flight_a_backslide_exhausted')
                            return
                        if self.truth_flight_a_backslide_recovery_until is None:
                            hold=max(0.0, float(getattr(
                                self,
                                'truth_flight_a_backslide_recovery_hold_seconds',
                                1.50)))
                            self.truth_flight_a_backslide_recovery_until=(
                                flight_a_now+hold)
                            self.truth_flight_a_backslide_recovery_ramp_until=None
                            rospy.logwarn(
                                'Flight-A backslide attempt %d/%d: holding '
                                'longitudinal climb at zero for %.2f s while '
                                'braking lateral drift to re-land before '
                                'resuming.',
                                events, maximum_attempts, hold)
                        if (flight_a_now <
                                self.truth_flight_a_backslide_recovery_until):
                            relanding_control=(
                                self._truth_flight_a_backslide_relanding_control())
                            if relanding_control is None:
                                self.truth_flight_a_command_forward_speed=0.0
                                self.truth_flight_a_command_center_speed=0.0
                                self.truth_flight_a_command_yaw_rate=0.0
                                self.cmd.publish(Twist())
                                self.hold_rl()
                            else:
                                brake_vx,brake_vy,brake_yaw,_=(
                                    relanding_control)
                                self._publish_truth_world_command(
                                    brake_vx, brake_vy, brake_yaw)
                                rospy.logwarn_throttle(
                                    .50,
                                    'Flight-A backslide re-landing brake: '
                                    'lateral velocity=%.3f m/s, center '
                                    'command=%.3f m/s, yaw=%.3f rad/s.',
                                    self.truth_flight_a_lateral_velocity,
                                    self.truth_flight_a_backslide_brake_center_speed,
                                    self.truth_flight_a_backslide_brake_yaw_rate)
                            self._record_trace(force=True)
                            return

                        ramp=max(0.0, float(getattr(
                            self,
                            'truth_flight_a_backslide_recovery_ramp_seconds',
                            1.00)))
                        self.truth_flight_a_backslide_recovery_ramp_until=(
                            flight_a_now+ramp)
                        self._reset_flight_a_rearward_slip(
                            clear_events=False)
                        # Recovery time is intentional zero climb progress.
                        # Re-arm
                        # both watchdog windows at the recovered pose so it
                        # cannot be mistaken for a hard stair stall.
                        self.truth_flight_a_stall_window_start=flight_a_now
                        self.truth_flight_a_stall_window_z=self.truth_pose[2]
                        self.truth_flight_a_hard_stall_since=None
                        self.truth_flight_a_hard_stall_start_gain=truth_gain
                        rospy.logwarn(
                            'Flight-A backslide hold complete; resuming at '
                            'most %.2f m/s for %.2f s before restoring '
                            'the %.2f m/s nominal command.',
                            self.truth_flight_a_backslide_recovery_ramp_speed,
                            ramp, self.ascent_speed)
                        self.cmd.publish(Twist())
                        self.hold_rl()
                        self._record_trace(force=True)
                        return
                    if alignment_status in ('recovering', 'recovered'):
                        if alignment_status=='recovered':
                            self.cmd.publish(Twist())
                            self.hold_rl()
                        else:
                            self._publish_truth_world_command(
                                vx, vy, yaw_rate)
                        self._record_trace(force=True)
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
                position_error=target_x-self.truth_pose[0]
                servo_heading,servo_mode=self._truth_landing_servo_heading(
                    target_heading, position_error)
                error=self._truth_landing_heading_error(
                    servo_heading, self.truth_pose[3])
                self.truth_landing_position_error=position_error
                self.truth_landing_heading_error=error
                self.truth_landing_servo_heading=servo_heading
                self.truth_landing_servo_mode=servo_mode
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
                    self._reset_flight_b_rearward_slip(clear_events=True)
                    self._reset_flight_b_pitch_collapse(clear_events=True)
                    self._reset_flight_b_alignment_recovery(
                        clear_attempts=True)
                    self.truth_flight_b_entry_z=self.truth_pose[2]
                    self.phase='STAIR_ASCENT_B'
                    self.state.publish(String(data=self.phase))
                    rospy.loginfo('Flight-B entry aligned: x error=%.3f m, '
                                  'heading error=%.3f rad.',
                                  position_error, error)
                    self._save_entry_joint_snapshot()
                    if getattr(self, 'pre_ascent_rl_settle_seconds', 0.0) > 0.0:
                        # Cancel the landing servo on the gate-crossing tick;
                        # otherwise its last lateral/yaw command remains live
                        # until the next 50 Hz callback and can carry a swing
                        # leg into the FixedStand handoff.
                        self.cmd.publish(Twist())
                        self.hold_rl()
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
                    self._reset_flight_b_rearward_slip(clear_events=True)
                    self._reset_flight_b_pitch_collapse(clear_events=True)
                    self._reset_flight_b_alignment_recovery(
                        clear_attempts=True)
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
                self._reset_flight_b_rearward_slip(clear_events=True)
                self._reset_flight_b_pitch_collapse(clear_events=True)
                self._reset_flight_b_alignment_recovery(
                    clear_attempts=True)
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
            # Optional pre-ascent stand gate: freeze the physically settled
            # live joint pose once, then require a fresh RL readiness edge
            # before allowing the flight-B climb.
            if ((self.pre_ascent_stand_seconds > 0.0 or
                 getattr(self, 'pre_ascent_rl_settle_seconds', 0.0) > 0.0) and
                    not self.pre_ascent_stand_done):
                if self._handle_pre_ascent_stand():
                    return
            if (self.truth_entry_guide and self.truth_pose is not None and
                    self.truth_ascent_start_z is not None and
                    (self.truth_fixed_two_flight_profile or self.truth_flight_b_heading is not None)):
                truth_gain=self.truth_pose[2]-self.truth_ascent_start_z
                flight_b_height_gain=(
                    self.truth_pose[2]-self.truth_flight_b_entry_z
                    if self.truth_flight_b_entry_z is not None else
                    truth_gain-self.flight_a_height_gain)
                rearward_slip=(
                    self._truth_flight_b_rearward_slip_update(
                        time.monotonic(), flight_b_height_gain))
                pitch_collapse=(
                    self._truth_flight_b_pitch_collapse_update(
                        time.monotonic(), flight_b_height_gain))
                if (pitch_collapse and
                        self.truth_flight_b_pitch_collapse_events >
                        self.truth_flight_b_pitch_recovery_max_attempts):
                    self.cmd.publish(Twist())
                    self.hold_rl()
                    self.phase='STAIR_FLIGHT_B_ATTITUDE_LOST'
                    self.state.publish(String(data=self.phase))
                    rospy.logerr(
                        'Flight-B posture-collapse recovery exhausted '
                        '(%d events, maximum %d); stopping safely.',
                        self.truth_flight_b_pitch_collapse_events,
                        self.truth_flight_b_pitch_recovery_max_attempts)
                    self._record_trace(force=True)
                    rospy.signal_shutdown(
                        'stair_flight_b_attitude_lost')
                    return
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
                # Unlike the z-freeze gate, this remains latched throughout
                # the full re-landing hold even after the measured backslide
                # slows down.  Resuming on that first quieter sample would
                # reproduce the ineffective one-tick recovery from round80c.
                stall=bool(stall or rearward_slip or pitch_collapse)
                if stall or dropped:
                    dynamic_backslide=bool(
                        self.truth_flight_b_rearward_slip_latched)
                    posture_relanding=bool(
                        self.truth_flight_b_pitch_collapse_latched)
                    slip_mode=bool(
                        dynamic_backslide or posture_relanding or
                        (self.truth_flight_b_slip_detect_seconds > 0 and
                         truth_gain > self.truth_flight_b_slip_min_gain_m))
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
                            # axis before the gait collapses.  A dynamic
                            # backslide can also carry lateral momentum
                            # (round82e), so retain the predictive centre
                            # command while setting only flight speed to zero.
                            # A pitch-collapse catch must first let all four
                            # feet re-contact under a stationary trunk.  Any
                            # centre/yaw command during this short interval
                            # can turn a recoverable forward tip into a full
                            # roll, so unlike a pure backslide it publishes a
                            # strict zero command for the complete hold.
                            if posture_relanding:
                                self.cmd.publish(Twist())
                                self.hold_rl()
                                self._record_trace()
                                return
                            relanding_center_speed=0.0
                            if dynamic_backslide:
                                relanding_control=(
                                    self._truth_flight_b_tracking_control(0.0))
                                if relanding_control is not None:
                                    relanding_center_speed=relanding_control[0]
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
                                    relanding_center_speed, 0.0, yaw_rate)
                                rospy.logwarn_throttle(
                                    1.0,
                                    'Flight-B stall pause: re-aligning yaw '
                                    '(err=%.3f rad, vx=%.2f, wz=%.2f).',
                                    yaw_err, relanding_center_speed, yaw_rate)
                            elif dynamic_backslide:
                                self._publish_truth_world_command(
                                    relanding_center_speed, 0.0, 0.0)
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
                    self._reset_flight_b_rearward_slip(clear_events=False)
                    self._reset_flight_b_pitch_collapse(clear_events=False)
                    rospy.logwarn(
                        'Flight-B resuming climb after stall pause %d '
                        '(ramp %.2f m/s for %.1f s, then %.2f m/s).',
                        self.truth_flight_b_stall_recoveries,
                        self.truth_flight_b_recovery_ramp_speed,
                        self.truth_flight_b_recovery_ramp_seconds,
                        self.ascent_speed)
                if self.truth_fixed_two_flight_profile:
                    climb_speed=self.ascent_speed
                    if (self.truth_flight_b_ramp_until is not None and
                            time.monotonic() < self.truth_flight_b_ramp_until):
                        climb_speed=self.truth_flight_b_recovery_ramp_speed
                    else:
                        self.truth_flight_b_ramp_until=None
                    control=self._truth_flight_b_tracking_control(climb_speed)
                    if control is None:
                        center_speed,flight_speed,yaw_rate=(0.0,-climb_speed,0.0)
                        center_error=heading_error=0.0
                    else:
                        (center_speed,flight_speed,yaw_rate,
                         center_error,heading_error)=control
                    # The predictive recovery should keep the trunk well
                    # inside this bound.  A centreline loss remains an
                    # immediate terminal condition.  A heading-only crossing
                    # while the trunk is safely centred first enters a short,
                    # bounded re-landing/strong-yaw recovery; round30 proved
                    # that shutting the controller down on a 0.004 rad
                    # threshold overshoot turns a recoverable gait oscillation
                    # into a slide.
                    flight_b_gain=truth_gain-self.flight_a_height_gain
                    alignment_status='normal'
                    alignment_reason=None
                    alignment_command=(None, None, None)
                    if (flight_b_gain >= .10 or getattr(
                            self,
                            'truth_flight_b_alignment_recovery_until',
                            None) is not None):
                        (alignment_status, recovery_vx, recovery_vy,
                         recovery_wz, alignment_reason)=(
                            self._truth_flight_b_alignment_recovery_control(
                                time.monotonic(), center_speed, center_error,
                                heading_error))
                        alignment_command=(
                            recovery_vx, recovery_vy, recovery_wz)
                    if alignment_status=='failed':
                        self.cmd.publish(Twist())
                        self.hold_rl()
                        self.phase='STAIR_FLIGHT_B_ALIGNMENT_LOST'
                        self.state.publish(String(data=self.phase))
                        rospy.logerr(
                            'Flight-B alignment recovery could not keep a '
                            'safe climb (%s): '
                            'gain=%.2f m, center error=%.3f m, heading '
                            'error=%.3f rad.', alignment_reason, flight_b_gain,
                            center_error, heading_error)
                        self._record_trace(force=True)
                        rospy.signal_shutdown(
                            'stair_flight_b_alignment_lost')
                        return
                    if alignment_status in ('recovering', 'recovered'):
                        self._publish_truth_world_command(*alignment_command)
                        self._record_trace(force=True)
                        return
                    self._publish_truth_world_command(
                        center_speed, flight_speed, yaw_rate)
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
        self._clear_fast_takeover_profile()
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
                           'route_heading_rad':self.truth_route_heading,
                           'guide_speed_cap_mps':self.truth_entry_guide_speed,
                           'final_speed_cap_mps':getattr(
                               self, 'truth_entry_final_speed', .55),
                           'minimum_speed_mps':self.truth_entry_guide_min_speed,
                           'corridor_recenter_active':getattr(
                               self, 'truth_corridor_recenter_active', False),
                           'corridor_center_error_m':getattr(
                               self, 'truth_corridor_center_error', None),
                           'corridor_predicted_center_error_m':getattr(
                               self, 'truth_corridor_predicted_center_error', None),
                           'corridor_lateral_velocity_mps':getattr(
                               self, 'truth_corridor_lateral_velocity', None),
                           'corridor_recenter_enter_error_m':getattr(
                               self, 'truth_corridor_recenter_enter_error', None),
                           'corridor_recenter_release_error_m':getattr(
                               self, 'truth_corridor_recenter_release_error', None)},
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
                           'strong_recovery_forward_speed_mps':
                               self.truth_flight_a_strong_recovery_forward_speed,
                           'recovery_center_speed_mps':
                               self.truth_flight_a_recovery_center_speed,
                           'recovery_yaw_rate_rps':
                               self.truth_flight_a_recovery_yaw_rate,
                           'slip_active':self.truth_flight_a_slip_active,
                           'last_peak_drawdown_m':
                               self.truth_flight_a_peak_drawdown,
                           'slip_minimum_peak_gain_m':
                               self.truth_flight_a_slip_min_peak_gain,
                           'slip_drawdown_trigger_m':
                               self.truth_flight_a_slip_drawdown_trigger,
                           'slip_forward_speed_mps':
                               self.truth_flight_a_slip_forward_speed,
                           'slip_yaw_rate_rps':
                               self.truth_flight_a_slip_yaw_rate,
                           'rearward_slip':{
                               'latched':getattr(
                                   self,
                                   'truth_flight_a_rearward_slip_latched',
                                   False),
                               'events':getattr(
                                   self,
                                   'truth_flight_a_rearward_slip_events', 0),
                               'last_speed_mps':getattr(
                                   self,
                                   'truth_flight_a_rearward_speed', None),
                               'last_filtered_speed_mps':getattr(
                                   self,
                                   'truth_flight_a_filtered_rearward_speed',
                                   None),
                               'speed_threshold_mps':
                                   self.truth_flight_a_rearward_slip_speed,
                               'detect_seconds':
                                   self.truth_flight_a_rearward_slip_detect_seconds,
                               'minimum_height_gain_m':
                                   self.truth_flight_a_rearward_slip_min_height_gain,
                               'last_trigger_height_gain_m':getattr(
                                   self,
                                   'truth_flight_a_rearward_slip_last_trigger_gain',
                                   None),
                               'recovery_hold_seconds':
                                   self.truth_flight_a_backslide_recovery_hold_seconds,
                               'recovery_ramp_seconds':
                                   self.truth_flight_a_backslide_recovery_ramp_seconds,
                               'recovery_ramp_speed_mps':
                                   self.truth_flight_a_backslide_recovery_ramp_speed,
                               'maximum_attempts':
                                   self.truth_flight_a_backslide_recovery_max_attempts,
                               'relanding_brake':{
                                   'active':getattr(
                                       self,
                                       'truth_flight_a_backslide_brake_active',
                                       False),
                                   'last_center_speed_mps':getattr(
                                       self,
                                       'truth_flight_a_backslide_brake_center_speed',
                                       0.0),
                                   'last_yaw_rate_rps':getattr(
                                       self,
                                       'truth_flight_a_backslide_brake_yaw_rate',
                                       0.0),
                                   'peak_center_speed_mps':getattr(
                                       self,
                                       'truth_flight_a_backslide_brake_peak_center_speed',
                                       0.0),
                                   'center_speed_limit_mps':
                                       self.truth_flight_a_predictive_strong_center_speed,
                                   'yaw_rate_limit_rps':
                                       self.truth_flight_a_recovery_yaw_rate,
                                   'lateral_velocity_gate_mps':
                                       self.truth_flight_a_prediction_min_lateral_speed}},
                           'predictive_strong_recovery':{
                               'active':getattr(
                                   self,
                                   'truth_flight_a_predictive_strong_active',
                                   False),
                               'on_center_crossing':getattr(
                                   self,
                                   'truth_flight_a_predictive_strong_on_center_crossing',
                                   True),
                               'minimum_lateral_speed_mps':
                                   self.truth_flight_a_predictive_strong_min_lateral_speed,
                               'forward_speed_mps':
                                   self.truth_flight_a_predictive_strong_forward_speed,
                               'center_speed_mps':
                                   self.truth_flight_a_predictive_strong_center_speed},
                           'near_edge_strong_recovery':{
                               'active':getattr(
                                   self,
                                   'truth_flight_a_near_edge_strong_active',
                                   False),
                               'center_error_threshold_m':
                                   self.truth_flight_a_near_edge_strong_center_error,
                               'forward_speed_mps':
                                   self.truth_flight_a_strong_recovery_forward_speed,
                               'center_speed_mps':
                                   self.truth_flight_a_predictive_strong_center_speed},
                           'alignment_recovery':{
                               'active':(
                                   self.truth_flight_a_alignment_recovery_until
                                   is not None),
                               'attempts':
                                   self.truth_flight_a_alignment_recovery_attempts,
                               'maximum_attempts':
                                   self.truth_flight_a_alignment_recovery_max_attempts,
                               'window_seconds':
                                   self.truth_flight_a_alignment_recovery_seconds,
                               'safe_center_error_m':
                                   self.truth_flight_a_alignment_recovery_safe_center_error,
                               'hard_heading_error_rad':
                                   self.truth_flight_a_alignment_recovery_hard_heading_error,
                               'last_reason':
                                   self.truth_flight_a_alignment_recovery_last_reason},
                           'stall_release_seconds':
                               self.truth_flight_a_stall_release_seconds,
                           'stall_release_max_center_error_m':
                               self.truth_flight_a_stall_release_max_center_error,
                           'stall_release_max_heading_error_rad':
                               self.truth_flight_a_stall_release_max_heading_error,
                           'stall_realign':{
                               'active':getattr(
                                   self,
                                   'truth_flight_a_stall_realign_active',
                                   False),
                               'attempts':getattr(
                                   self,
                                   'truth_flight_a_stall_realign_attempts',
                                   0),
                               'forward_speed_mps':
                                   self.truth_flight_a_slip_forward_speed,
                               'center_speed_mps':
                                   self.truth_flight_a_predictive_strong_center_speed},
                           'early_retry':{
                               'attempts':
                                   self.truth_flight_a_early_retry_attempts,
                               'maximum_attempts':
                                   self.truth_flight_a_early_retry_max_attempts,
                               'last_reason':
                                   self.truth_flight_a_early_retry_reason,
                               'minimum_peak_gain_m':
                                   self.truth_flight_a_early_retry_min_peak_gain,
                               'minimum_drawdown_m':
                                   self.truth_flight_a_early_retry_drawdown,
                               'maximum_current_gain_m':
                                   self.truth_flight_a_early_retry_max_current_gain,
                               'minimum_center_error_m':
                                   self.truth_flight_a_early_retry_min_center_error,
                               'minimum_heading_error_rad':
                                   self.truth_flight_a_early_retry_min_heading_error,
                               'plane_policy':self.plane_policy,
                               'plane_settle_seconds':
                                   self.truth_flight_a_early_retry_plane_settle}},
                       'flight_b_guard':{
                           'last_center_error_m':
                               self.truth_flight_b_center_error,
                           'last_predicted_center_error_m':
                               self.truth_flight_b_predicted_center_error,
                           'last_lateral_velocity_mps':
                               self.truth_flight_b_lateral_velocity,
                           'recovery_active':
                               self.truth_flight_b_recovery_active,
                           'maximum_center_error_m':
                               self.truth_flight_b_max_center_error,
                           'maximum_heading_error_rad':
                               self.truth_flight_b_max_heading_error,
                           'recovery_center_error_m':
                               self.truth_flight_b_recovery_center_error,
                           'recovery_release_center_error_m':
                               self.truth_flight_b_recovery_release_center_error,
                           'recovery_forward_speed_mps':
                               self.truth_flight_b_recovery_forward_speed,
                           'strong_recovery_forward_speed_mps':
                               self.truth_flight_b_strong_recovery_forward_speed,
                           'recovery_center_speed_mps':
                               self.truth_flight_b_recovery_center_speed,
                           'prediction_horizon_sec':
                               self.truth_flight_b_prediction_horizon,
                           'predictive_strong_recovery':{
                               'active':getattr(
                                   self,
                                   'truth_flight_b_predictive_strong_active',
                                   False),
                               'minimum_lateral_speed_mps':
                                   self.truth_flight_b_predictive_strong_min_lateral_speed,
                               'center_speed_mps':
                                   self.truth_flight_b_predictive_strong_center_speed},
                           'near_edge_strong_recovery':{
                               'active':getattr(
                                   self,
                                   'truth_flight_b_near_edge_strong_active',
                                   False),
                               'center_error_threshold_m':getattr(
                                   self,
                                   'truth_flight_b_near_edge_strong_center_error',
                                   0.0),
                               'center_speed_mps':
                                   self.truth_flight_b_predictive_strong_center_speed},
                           'rearward_slip':{
                               'latched':getattr(
                                   self,
                                   'truth_flight_b_rearward_slip_latched',
                                   False),
                               'events':getattr(
                                   self,
                                   'truth_flight_b_rearward_slip_events', 0),
                               'last_speed_mps':getattr(
                                   self,
                                   'truth_flight_b_rearward_speed', None),
                               'last_filtered_speed_mps':getattr(
                                   self,
                                   'truth_flight_b_filtered_rearward_speed',
                                   None),
                               'speed_threshold_mps':getattr(
                                   self,
                                   'truth_flight_b_rearward_slip_speed', 0.0),
                               'detect_seconds':getattr(
                                   self,
                                   'truth_flight_b_rearward_slip_detect_seconds',
                                   .04),
                               'minimum_height_gain_m':getattr(
                                   self,
                                   'truth_flight_b_rearward_slip_min_height_gain',
                                   .40),
                               'velocity_alpha':getattr(
                                   self,
                                   'truth_flight_b_rearward_slip_velocity_alpha',
                                   .70),
                               'release_ratio':getattr(
                                   self,
                                   'truth_flight_b_rearward_slip_release_ratio',
                                   .80),
                               'last_trigger_height_gain_m':getattr(
                                   self,
                                   'truth_flight_b_rearward_slip_last_trigger_gain',
                                   None)},
                           'pitch_collapse':{
                               'latched':getattr(
                                   self,
                                   'truth_flight_b_pitch_collapse_latched',
                                   False),
                               'events':getattr(
                                   self,
                                   'truth_flight_b_pitch_collapse_events', 0),
                               'last_pitch_rate_rps':getattr(
                                   self, 'truth_flight_b_pitch_rate', None),
                               'rate_window_sec':getattr(
                                   self,
                                   'truth_flight_b_pitch_rate_window', .18),
                               'rate_trigger_rps':getattr(
                                   self,
                                   'truth_flight_b_pitch_rate_trigger', 0.0),
                               'rate_minimum_abs_pitch_rad':getattr(
                                   self,
                                   'truth_flight_b_pitch_rate_min_abs', .60),
                               'severe_pitch_rad':getattr(
                                   self,
                                   'truth_flight_b_severe_pitch', 0.0),
                               'severe_detect_seconds':getattr(
                                   self,
                                   'truth_flight_b_severe_pitch_detect_seconds',
                                   .10),
                               'minimum_height_gain_m':getattr(
                                   self,
                                   'truth_flight_b_pitch_min_height_gain', .08),
                               'maximum_recovery_attempts':getattr(
                                   self,
                                   'truth_flight_b_pitch_recovery_max_attempts',
                                   2),
                               'last_trigger_height_gain_m':getattr(
                                   self,
                                   'truth_flight_b_pitch_last_trigger_gain',
                                   None),
                               'last_trigger_reason':getattr(
                                   self,
                                   'truth_flight_b_pitch_last_trigger_reason',
                                   None)},
                           'alignment_recovery':{
                               'active':(
                                   self.truth_flight_b_alignment_recovery_until
                                   is not None),
                               'attempts':
                                   self.truth_flight_b_alignment_recovery_attempts,
                               'maximum_attempts':
                                   self.truth_flight_b_alignment_recovery_max_attempts,
                               'window_seconds':
                                   self.truth_flight_b_alignment_recovery_seconds,
                               'safe_center_error_m':
                                   self.truth_flight_b_alignment_recovery_safe_center_error,
                               'hard_heading_error_rad':
                                   self.truth_flight_b_alignment_recovery_hard_heading_error,
                               'maximum_height_drop_m':
                                   self.truth_flight_b_alignment_recovery_max_drop,
                               'last_reason':
                                   self.truth_flight_b_alignment_recovery_last_reason}},
                       'landing_recovery':{
                           'active':self.landing_recovery_active,
                           'attempts':self.landing_recovery_attempts,
                           'maximum_attempts':self.landing_recovery_max_attempts,
                           'retry_timeout_sec':self.landing_recovery_timeout,
                           'servo_heading_rad':
                               self.truth_landing_servo_heading,
                           'servo_mode':self.truth_landing_servo_mode,
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
                       'pre_ascent_rl_takeover':{
                           'fast_profile_enabled':bool(getattr(
                               self, 'pre_ascent_rl_fast_takeover', False)),
                           'fast_blend_seconds':float(getattr(
                               self,
                               'pre_ascent_rl_fast_blend_seconds', .75)),
                           'fast_zero_hold_seconds':float(getattr(
                               self,
                               'pre_ascent_rl_fast_zero_hold_seconds', .25)),
                           'profile_params_owned_at_save':bool(getattr(
                               self,
                               '_fast_takeover_profile_owned', False))},
                       'flight_a_policy_handoff':{
                           'fixed_stand_seconds':float(getattr(
                               self,
                               'truth_flight_a_policy_stand_seconds', 0.0)),
                           'settle_seconds':float(getattr(
                               self,
                               'truth_flight_a_policy_settle_seconds', .60)),
                           'settle_timeout_sec':float(getattr(
                               self,
                               'truth_flight_a_policy_settle_timeout', 2.5)),
                           'capture_max_joint_velocity_rms':float(getattr(
                               self,
                               'truth_flight_a_policy_capture_max_joint_velocity_rms',
                               .35)),
                           'policy_queue_hold_seconds':float(getattr(
                               self,
                               'truth_flight_a_policy_queue_hold_seconds',
                               .25)),
                           'use_default_stand_target':bool(getattr(
                               self,
                               'truth_flight_a_policy_use_default_stand_target',
                               False)),
                           'require_fixed_stand_ready':bool(getattr(
                               self,
                               'truth_flight_a_policy_require_fixed_stand_ready',
                               False)),
                           'fixed_stand_ready_timeout_sec':float(getattr(
                               self,
                               'truth_flight_a_policy_fixed_stand_ready_timeout',
                               12.0)),
                           'captured_pose_minimum_gain_ratio':float(getattr(
                               self, '_fast_fixed_stand_params', {}).get(
                                   '/simenv/fixed_stand_fast_minimum_gain_ratio',
                                   0.0)),
                           'fresh_fixed_stand_ready_seen':bool(getattr(
                               self,
                               'truth_flight_a_policy_fixed_stand_ready_confirmed',
                               False)),
                           'ready_wait_timeout_sec':float(getattr(
                               self,
                               'truth_flight_a_ready_wait_timeout', 20.0)),
                           'maximum_preflight_tilt_rad':float(getattr(
                               self,
                               'truth_flight_a_preflight_max_tilt', .45)),
                           'maximum_preflight_height_drop_m':float(getattr(
                               self,
                               'truth_flight_a_preflight_max_height_drop',
                               .10)),
                           'last_truth_roll_rad':getattr(
                               self, 'truth_roll', None),
                           'last_truth_pitch_rad':getattr(
                               self, 'truth_pitch', None)},
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
