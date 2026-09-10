#!/usr/bin/env python3
"""F3 探索完成后的楼梯下行返程管理器(F3 → F2 → F1)。

在 F3 探索完成(THIRD_FLOOR_EXPLORATION_COMPLETE)后接管,沿原双跑楼梯
下行返回一楼。每段下梯(与上梯相反,先 flight_b 后 flight_a):

  DESCENT_ENTRY_GUIDE(仅段 1:F3 走廊口 → landing 侧点 → landing 中心)
    → STAIR_DESCENT_PRE_ALIGN(原地对正 flight_b 下行朝向)
    → STAIR_DESCENT_POLICY_LOADING(请求 stair 策略,等 policy_reloaded)
    → STAIR_DESCENT_POLICY_WARMUP(短等待)
    → STAIR_DESCENT_STAND(8 s)
    → STAIR_DESCENT_FLIGHT_B(沿 +y 下行,drop ≥1.05 m 且 y ≥4.70)
    → STAIR_DESCENT_TURN(中间平台:转到 -pi/2 + 横移到 flight_a 中心线)
    → STAIR_DESCENT_FLIGHT_A(沿 -y 下行,drop ≥2.45 m 且 y ≤2.25)
    → 到达下一层 landing
  段 1(F3→F2)在 F2 landing 上原地 180° 转体(STAIR_DESCENT_SEGMENT_TURN),
  继续段 2(F2→F1)直至 FIRST_FLOOR_RETURNED。

控制方案与上梯 stair_transition_manager 同源:stair 策略 + 世界系命令
(_publish_truth_world_command),flight 上做中心线/航向伺服;负向速度
(后退 backoff)已在 F2→F3 flight-b 恢复中生产验证。

产出(统计兼容):
  logs/stair_descent.json                           全程主标志(phase/trace)
  logs/third_to_second_floor_stair_transition.json  段 1 终态
  logs/second_to_first_floor_stair_transition.json  段 2 终态
  {output_dir}/first_floor_returned.json            全程完成标志
"""
import json, math, os, time, shutil, subprocess, sys, threading
from collections import deque
import xml.etree.ElementTree as ET
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Joy, JointState
from std_msgs.msg import String, Bool
from gazebo_msgs.msg import ModelStates, LinkStates, ModelState
from gazebo_msgs.srv import SetModelState
from tf.transformations import euler_from_quaternion

def yaw(q):
    return math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))

class StairDescent:
    def __init__(self):
        self.out=os.path.abspath(rospy.get_param('~output_dir'))
        self.policy=rospy.get_param('~stair_policy')
        self.start_floor_number=int(rospy.get_param('~start_floor_number', 3))
        self.end_floor_number=int(rospy.get_param('~end_floor_number', 1))
        # 当前段使用的楼梯链接楼层号:段 1(F3→F2)用 floor_1,段 2(F2→F1)用 floor_0
        self.flight_floor_index=int(rospy.get_param(
            '~flight_floor_index', self.start_floor_number-2))
        self.skip_entry_guide=bool(rospy.get_param('~skip_entry_guide', False))
        self.mission_trigger_topic=str(rospy.get_param(
            '~mission_trigger_topic', '')).strip()
        self.mission_trigger_token=str(rospy.get_param(
            '~mission_trigger_token', '')).strip()
        self.mission_trigger_tokens={
            token.strip() for token in str(rospy.get_param(
                '~mission_trigger_tokens',
                self.mission_trigger_token)).split(',') if token.strip()}
        self.mission_failure_tokens={
            token.strip() for token in str(rospy.get_param(
                '~mission_failure_tokens',
                'THIRD_FLOOR_EXPLORATION_FAILED')).split(',')
            if token.strip()}
        self.terminate_on_mission_failure=bool(rospy.get_param(
            '~terminate_on_mission_failure', False))
        self.state_topic=str(rospy.get_param(
            '~state_topic', '/simenv/third_to_first_floor_stair_state'))
        self.transition_log_basename=str(rospy.get_param(
            '~transition_log_basename', 'stair_descent.json'))
        self.segment_one_log_basename=str(rospy.get_param(
            '~segment_one_log_basename',
            'third_to_second_floor_stair_transition.json'))
        self.segment_two_log_basename=str(rospy.get_param(
            '~segment_two_log_basename',
            'second_to_first_floor_stair_transition.json'))
        # 需下行的总段数(start_floor-end_floor):全流程 3→1 = 2 段,
        # 隔离 harness(F3→F2 或 F2→F1)= 1 段
        self.total_segments=max(1, self.start_floor_number-self.end_floor_number)
        self.phase='WAIT_F3'; self.pose=None; self.truth_pose=None
        self.truth_attitude=None
        self.started=None; self.trigger_wall_time=None
        self.segment_started=None
        self._boot_monotonic=time.monotonic()
        # /clock may still be zero while Gazebo starts.  Latch the first
        # positive ROS time in WAIT_F3 so the normal mission-trigger budget is
        # measured in the same simulation-time domain as exploration.
        self._boot_ros_time=None
        self.trace=[]; self.last_log=0.
        self.policy_loaded=False; self.locomotion_ready=False
        self.locomotion_ready_at=None
        self.fixed_stand_ready=False
        self.fixed_stand_ready_at=None
        self.fixed_stand_not_ready_at=None
        self.odom_seen=False; self.policy_requested=False
        self.segment_one_elapsed_sec=None
        self._pause_active=False
        self.trigger_ros_time=None
        self.pending_mission_failure_token=None
        self.pending_mission_failure_wall_time=None
        self.persisted_failure_poll_wall_time=0.0
        self.mission_failure_return_wait_timeout=float(rospy.get_param(
            '~mission_failure_return_wait_wall_timeout_sec', 45.0))
        # PRE_ALIGN used to have no deadline at all.  run133 consequently
        # remained on the F3 landing indefinitely after a failed mission,
        # publishing wz=0.4 while truth yaw was stationary.  Bound the phase
        # in both simulation and wall time and permit exactly one physical
        # re-planting arc before declaring a clear terminal failure.
        self.pre_align_timeout_ros_sec=max(8.0, float(rospy.get_param(
            '~truth_descent_pre_align_timeout_sec', 20.0)))
        self.pre_align_wall_timeout_sec=max(20.0, float(rospy.get_param(
            '~truth_descent_pre_align_wall_timeout_sec', 60.0)))
        self.pre_align_stall_ros_sec=max(1.5, float(rospy.get_param(
            '~truth_descent_pre_align_stall_sec', 3.0)))
        self.pre_align_minimum_yaw_rate=max(0.55, min(0.72, float(
            rospy.get_param('~truth_descent_pre_align_minimum_yaw_rate_rps',
                            0.65))))
        self.pre_align_started_ros=None
        self.pre_align_started_wall=None
        self.pre_align_progress_ros=None
        self.pre_align_progress_yaw=None
        self.pre_align_arc_until_ros=None
        self.pre_align_arc_used=False
        self.lower_floor_failure_return_mode=None
        self.lower_floor_policy_loading_started=None
        self.lower_floor_policy_last_request=None
        self.lower_floor_policy_request_count=0
        self.lower_floor_policy_retry_period=max(0.5, float(rospy.get_param(
            '~lower_floor_failure_policy_retry_period_sec', 2.0)))
        self.lower_floor_policy_timeout=max(2.0, float(rospy.get_param(
            '~lower_floor_failure_policy_timeout_sec', 12.0)))
        self.lower_floor_policy_max_requests=max(1, int(rospy.get_param(
            '~lower_floor_failure_policy_max_requests', 4)))
        self.first_floor_failure_return_maximum_z=float(rospy.get_param(
            '~first_floor_failure_return_maximum_z_m', 1.20))
        self.second_floor_failure_return_minimum_z=float(rospy.get_param(
            '~second_floor_failure_return_minimum_z_m', 2.30))
        self.second_floor_failure_return_maximum_z=float(rospy.get_param(
            '~second_floor_failure_return_maximum_z_m', 3.40))
        self.minimum_third_floor_failure_wait_height=float(rospy.get_param(
            '~minimum_third_floor_failure_wait_height_m', 4.80))
        self.third_floor_failure_landing_trigger_radius=float(rospy.get_param(
            '~third_floor_failure_landing_trigger_radius_m', 1.25))
        self.first_floor_landing_ros_time=None
        self.milestones={}
        self.post_trigger_model_state_reset_count=0
        self.top_lip_minimum_drop=float(rospy.get_param(
            '~top_lip_minimum_drop_m', 0.12))
        self.top_lip_minimum_advance=float(rospy.get_param(
            '~top_lip_minimum_advance_m', 0.30))
        self.milestone_maximum_tilt=float(rospy.get_param(
            '~milestone_maximum_tilt_rad', 0.35))
        # A top-lip milestone may legitimately be observed while the trunk is
        # pitched with the stairs.  Segment completion is different: the next
        # action is an in-place 180 degree turn, so accepting the final tread
        # as a landing can roll the dog off the edge.  run31 was accepted at
        # z=3.058 m / pitch=0.354 rad and subsequently overturned, whereas the
        # successful run28 reached the F2 slab at z=2.923 m / pitch=0.027 rad.
        # Keep a separate, stricter flat-landing attitude gate.
        self.landing_maximum_tilt=float(rospy.get_param(
            '~landing_maximum_tilt_rad', 0.14))
        self.second_floor_landing_maximum_z=float(rospy.get_param(
            '~second_floor_landing_maximum_z_m', 3.00))
        self.first_floor_landing_maximum_z=float(rospy.get_param(
            '~first_floor_landing_maximum_z_m', 0.48))
        self.landing_gate_stable_seconds=float(rospy.get_param(
            '~landing_gate_stable_ros_seconds', 0.50))
        self.landing_gate_stable_since_ros=None

        # ---- 一楼 landing -> 原始出生点 ----
        self.return_to_start_enabled=bool(rospy.get_param(
            '~return_to_first_floor_start', True))
        self.plane_policy=rospy.get_param('~plane_policy', '')
        self.home_x=float(rospy.get_param('~first_floor_start_x', 0.0))
        self.home_y=float(rospy.get_param('~first_floor_start_y', 0.8))
        self.home_yaw=float(rospy.get_param(
            '~first_floor_start_yaw', math.pi/2.0))
        # Leave the enclosed stair landing through its east opening first.
        # The south edge is a solid stair_core wall (y≈0.94); commanding a
        # same-x south waypoint stalls physically at y≈1.34.  x=-0.8 is on
        # the connected F1 slab east of that wall, after which the home goal
        # can be approached diagonally without crossing the stair enclosure.
        self.home_clear_x=float(rospy.get_param(
            '~first_floor_stair_clear_x', -0.80))
        self.home_clear_y=float(rospy.get_param(
            '~first_floor_stair_clear_y', 1.55))
        self.home_speed=float(rospy.get_param(
            '~first_floor_home_return_speed_mps', 0.85))
        self.home_minimum_speed=float(rospy.get_param(
            '~first_floor_home_return_minimum_speed_mps', 0.25))
        self.home_yaw_rate=float(rospy.get_param(
            '~first_floor_home_return_yaw_rate_rps', 0.65))
        self.home_position_tolerance=float(rospy.get_param(
            '~first_floor_home_position_tolerance_m', 0.25))
        self.home_heading_tolerance=float(rospy.get_param(
            '~first_floor_home_heading_tolerance_rad', 0.20))
        self.home_return_timeout=float(rospy.get_param(
            '~first_floor_home_return_wall_timeout_sec', 240.0))
        self.home_progress_timeout=float(rospy.get_param(
            '~first_floor_home_progress_wall_timeout_sec', 12.0))
        self.home_progress_threshold=float(rospy.get_param(
            '~first_floor_home_progress_threshold_m', 0.12))
        self.home_arrival_stable_seconds=float(rospy.get_param(
            '~first_floor_home_arrival_stable_ros_seconds', 1.0))
        self.home_minimum_upright_z=float(rospy.get_param(
            '~first_floor_home_minimum_upright_z_m', 0.20))
        self.home_maximum_tilt=float(rospy.get_param(
            '~first_floor_home_maximum_tilt_rad', 0.65))
        self.home_return_started=None
        self.home_return_ros_started=None
        self.home_stage='clear_stair'
        self.home_best_distance=None
        self.home_last_progress_at=None
        self.home_arrival_stable_since_ros=None
        self.home_fall_recovery_timeout=max(5.0, float(rospy.get_param(
            '~first_floor_home_fall_recovery_wall_timeout_sec', 35.0)))
        self.home_fall_recovery_max_attempts=max(0, int(rospy.get_param(
            '~first_floor_home_fall_recovery_max_attempts', 1)))
        self.home_fall_recovery_attempts=0
        self.home_fall_recovery_started=None
        self.home_fall_recovery_stable_since=None
        self.home_failure_corridor_x=float(rospy.get_param(
            '~first_floor_failure_corridor_center_x', 0.0))
        self.home_failure_corridor_return_y=float(rospy.get_param(
            '~first_floor_failure_corridor_return_y', 1.55))
        self.home_failure_egress_y=None
        self.home_failure_egress_recoveries=0
        self.home_failure_egress_max_recoveries=max(0, int(rospy.get_param(
            '~first_floor_failure_egress_max_recoveries', 1)))
        self.active_policy=self.policy
        self.visualization_script=str(rospy.get_param(
            '~roundtrip_visualization_script', '')).strip()
        self.combined_visualization_script=str(rospy.get_param(
            '~combined_visualization_script', '')).strip()
        self.visualization_launcher=str(rospy.get_param(
            '~visualization_after_shutdown_script', '')).strip()

        # ---- 下梯速度与超时 ----
        self.descent_speed=float(rospy.get_param('~descent_speed_mps', .35))
        self.descent_maximum_speed=float(rospy.get_param(
            '~descent_maximum_speed_mps', .40))
        self.descent_speed=min(self.descent_speed,
                               self.descent_maximum_speed)
        self.descent_segment_timeout=float(rospy.get_param(
            '~descent_segment_timeout_sec', 180.0))
        # Bounded truth seam correction for the pathological case where the
        # locomotion controller advances across the stair opening while the
        # Gazebo base height remains on the upper floor.  This is not ordinary
        # stall (y keeps changing), so the old z+y freeze detector never
        # fired and one descent consumed >120 s without dropping a step.
        self.truth_no_drop_correction_enabled=bool(rospy.get_param(
            '~truth_descent_no_drop_correction_enabled', True))
        self.truth_no_drop_advance=float(rospy.get_param(
            '~truth_descent_no_drop_advance_m', 0.85))
        self.truth_no_drop_max_drop=float(rospy.get_param(
            '~truth_descent_no_drop_max_drop_m', 0.04))
        self.truth_no_drop_max_corrections=int(rospy.get_param(
            '~truth_descent_no_drop_max_corrections', 1))
        self.truth_fall_recovery_enabled=bool(rospy.get_param(
            '~truth_descent_fall_recovery_enabled', True))
        self.truth_fall_recovery_max_attempts=max(0, int(rospy.get_param(
            '~truth_descent_fall_recovery_max_attempts', 1)))
        self.flight_fall_recoveries=0
        self.fall_recovery_count=0
        self.truth_partial_lip_max_drop=float(rospy.get_param(
            '~truth_descent_partial_lip_max_drop_m', 0.35))
        self.truth_partial_lip_stall_trigger=int(rospy.get_param(
            '~truth_descent_partial_lip_stall_trigger', 2))
        self.truth_no_drop_corrections=0
        self.trigger_timeout=float(rospy.get_param(
            '~truth_descent_trigger_timeout_sec', 300.0))
        # Separate dead-process protection.  This is intentionally much more
        # generous than the ROS-time mission budget: low RTF must not kill a
        # healthy outbound mission, while a frozen /clock still terminates in
        # bounded wall time.
        self.trigger_wall_timeout=float(rospy.get_param(
            '~truth_descent_trigger_wall_timeout_sec', 14400.0))
        # 后向下梯:目标朝向 = 各 flight 的上行方向(背对下行方向),
        # 世界系命令不变,投影自动产生 body 负向速度(与 flight-b backoff 同源)。
        self.backward_mode=bool(rospy.get_param(
            '~truth_descent_backward_mode', False))
        self.pre_descent_stand_seconds=float(rospy.get_param(
            '~pre_descent_stand_seconds', 8.0))
        self.pre_descent_stand_started=None
        self.pre_descent_stand_done=False
        # Preserve the proven overall_l0829 plane->stair controller boundary.
        # A direct hot reload can feed one plane-history sample to the stair
        # policy and leave the robot shuffling on the upper lip (run93).
        self.policy_switch_settle_seconds=max(0.0, float(rospy.get_param(
            '~policy_switch_settle_seconds', .60)))
        self.policy_switch_settle_timeout=max(
            self.policy_switch_settle_seconds,
            float(rospy.get_param('~policy_switch_settle_timeout_sec', 6.0)))
        self.policy_switch_joint_velocity_rms=max(0.0, float(rospy.get_param(
            '~policy_switch_max_joint_velocity_rms', .35)))
        # A low-velocity single gait frame is not necessarily load-bearing:
        # run176 captured an asymmetric zero-command trot phase and FixedStand
        # immediately tipped the robot over.  Average a complete stable window
        # before constructing the stand target.
        self.policy_switch_snapshot_window=max(.35, float(rospy.get_param(
            '~policy_switch_snapshot_window_sec', .60)))
        self.policy_switch_snapshot_min_samples=max(3, int(rospy.get_param(
            '~policy_switch_snapshot_min_samples', 8)))
        self.policy_switch_fixed_stand_timeout=max(0.1, float(rospy.get_param(
            '~policy_switch_fixed_stand_timeout_sec', 12.0)))
        self.policy_switch_queue_hold_seconds=max(0.05, float(rospy.get_param(
            '~policy_switch_queue_hold_seconds', .25)))
        self.policy_switch_reentry_timeout=max(0.1, float(rospy.get_param(
            '~policy_switch_reentry_timeout_sec', 20.0)))
        self.policy_switch_settle_started=None
        self.policy_switch_stable_since=None
        self.policy_switch_snapshot_started=None
        self.policy_switch_stand_started=None
        self.policy_switch_queued_at=None
        self.policy_switch_reentry_started=None
        self._stand_target_param='/stand_target_joints'
        self._stand_target_applied=False
        self.stand_target_evidence=None
        self._fast_takeover_params={
            '/simenv/stair_fast_takeover_blend_seconds':.75,
            '/simenv/stair_fast_takeover_zero_hold_seconds':.25,
            '/simenv/stair_fast_takeover_enabled':True}
        self._fast_takeover_profile_owned=False
        self.policy_switch_fast_fixed_stand=bool(rospy.get_param(
            '~policy_switch_fast_fixed_stand', True))
        self._fast_fixed_stand_params={
            '/simenv/fixed_stand_fast_duration_seconds':float(rospy.get_param(
                '~policy_switch_fast_fixed_stand_duration_seconds', 1.0)),
            '/simenv/fixed_stand_fast_settle_seconds':float(rospy.get_param(
                '~policy_switch_fast_fixed_stand_settle_seconds', .10)),
            '/simenv/fixed_stand_fast_minimum_elapsed_seconds':float(
                rospy.get_param(
                    '~policy_switch_fast_fixed_stand_minimum_elapsed_seconds',
                    1.10)),
            '/simenv/fixed_stand_fast_stable_seconds':float(rospy.get_param(
                '~policy_switch_fast_fixed_stand_stable_seconds', 1.0)),
            '/simenv/fixed_stand_fast_minimum_gain_ratio':min(
                1.0, max(0.0, float(rospy.get_param(
                    '~policy_switch_fast_fixed_stand_minimum_gain_ratio',
                    0.40)))),
            '/simenv/fixed_stand_fast_ready_enabled':True}
        self._fast_fixed_stand_profile_owned=False
        self._joint_lock=threading.RLock()
        self.joint_names=None
        self.joint_positions=None
        self.joint_velocity_rms=None
        self.joint_received_at=None
        self.joint_samples=deque(maxlen=512)
        self.truth_twist=None
        self.landing_gate_stable_since_ros=None

        # ---- 入口引导(F3 走廊口 → F3 landing)----
        self.truth_entry_guide=bool(rospy.get_param('~truth_entry_guide', True))
        self.truth_two_stage_guide=bool(rospy.get_param(
            '~truth_two_stage_guide', True))
        self.offline_stair_model=rospy.get_param('~offline_stair_model_sdf', '')
        self.offline_layout=rospy.get_param('~offline_truth_layout_metadata', '')
        self.truth_entry_guide_speed=float(rospy.get_param(
            '~truth_descent_entry_guide_speed_mps', .90))
        self.truth_entry_guide_min_speed=float(rospy.get_param(
            '~truth_descent_entry_guide_min_speed_mps', .25))
        self.truth_entry_timeout=float(rospy.get_param(
            '~truth_descent_entry_timeout_sec', 90.0))
        self.truth_entry_deadline=None
        self.truth_entry_watchdog_anchor_distance=None
        self.truth_entry_watchdog_progress=float(rospy.get_param(
            '~truth_descent_entry_watchdog_progress_m', .25))
        self.truth_side_entry_offset=float(rospy.get_param(
            '~truth_descent_side_entry_offset_m', 1.65))
        self.truth_side_approach_radius=float(rospy.get_param(
            '~truth_descent_side_approach_radius_m', 6.0))
        self.truth_corridor_lateral_offset=float(rospy.get_param(
            '~truth_descent_corridor_lateral_offset_m', 1.80))
        self.truth_corridor_longitudinal_offset=float(rospy.get_param(
            '~truth_descent_corridor_longitudinal_offset_m', 3.80))
        self.truth_upper_floor_exit_clearance=float(rospy.get_param(
            '~truth_descent_upper_floor_exit_clearance_m', .80))
        self.entry_alignment_yaw_rate=float(rospy.get_param(
            '~descent_entry_alignment_yaw_rate_rps', .40))
        self.truth_entry_stage='corridor'
        self.truth_corridor_target=None
        self.truth_side_target=None
        self.truth_entry_target=None
        self.truth_route_heading=None
        self.truth_entry_last_distance=None
        self.truth_entry_best_distance=None
        self.truth_entry_stage_switches=0
        self.truth_using_corridor_return=False
        self.truth_landing_center=None      # 入口引导 final 目标(landing 中心)
        self.truth_landing_heading=None     # 入口引导最终朝向(flight_b 下行方向)

        # ---- 楼梯几何(LinkStates 解析)----
        self.truth_step_pose=None           # flight_a step_0
        self.truth_step_next_pose=None      # flight_a step_1
        self.truth_flight_a_top=None        # flight_a step_9
        self.truth_flight_a_heading=None    # flight_a 上行方向(沿 step0->step1)
        self.truth_flight_b_pose=None       # flight_b step_0
        self.truth_flight_b_next_pose=None  # flight_b step_1
        self.truth_flight_b_top=None        # flight_b step_9
        self.truth_flight_b_heading=None    # flight_b 上行方向
        self._geometry_dirty=True

        # ---- 下梯守卫参数 ----
        self.stall_seconds=float(rospy.get_param(
            '~truth_descent_stall_seconds', 4.0))
        self.stall_pause_seconds=float(rospy.get_param(
            '~truth_descent_stall_pause_seconds', 1.0))
        self.stall_z_progress=float(rospy.get_param(
            '~truth_descent_stall_z_progress_m', .02))
        self.stall_min_elapsed=float(rospy.get_param(
            '~truth_descent_stall_min_elapsed_sec', 2.0))
        # 碎步停顿:每下阶梯间 hold 片刻,让躯干/腿部姿态稳定再继续。
        # 冒烟 4 连败的共性 = 第一步能下(0.16-0.25m),第二步栽倒;
        # 连续下行时步态来不及收敛。0 值禁用。
        self.step_pause_seconds=float(rospy.get_param(
            '~truth_descent_step_pause_seconds', 0.0))
        self.step_pause_interval_m=float(rospy.get_param(
            '~truth_descent_step_pause_interval_m', .26))
        self.last_step_pause_drop=None
        # stall 守卫武装门槛:机器人必须先真正开始下梯(drop 达到第一步)
        # 才启用卡死检测。平坦 landing 上走向楼梯时 drop 恒为 0,若此刻
        # 武装会把"还没走到楼梯"误判为卡死,backoff 将机器人拉回原点
        # (backward-0.20 冒烟:20 次 stall,0 前进,SEGMENT_TIMEOUT)。
        self.stall_arm_drop=float(rospy.get_param(
            '~truth_descent_stall_arm_drop_m', .05))
        # fixV4(2026-08-25):stall 武装盲区 —— 段2 FLIGHT_B 从 F2 landing
        # 起步即卡时 drop≈0 永不武装(iter1 run3 实证 z=2.74 卡死 577s
        # 无恢复)。沿下梯方向前进 ≥stall_y_arm 视为已上楼梯,同等武装;
        # 卡死判定在 z 冻结基础上加 y 冻结(窗口内 y 前进 < 阈值)。
        self.stall_y_arm=float(rospy.get_param(
            '~truth_descent_stall_y_arm_m', .50))
        self.stall_y_progress=float(rospy.get_param(
            '~truth_descent_stall_y_progress_m', .04))
        self.flight_start_y=None
        self.backoff_distance=float(rospy.get_param(
            '~truth_descent_backoff_distance_m', .35))
        self.backoff_speed=float(rospy.get_param(
            '~truth_descent_backoff_speed_mps', .30))
        # 2026-08-15 v2:backoff 距离随 stall 次数升级 + 后退期间
        # yaw 伺服上限。v1 固定 0.35m 每轮只退到上一步,重新起步
        # 又卡同一级 riser(stall #1-#9 原地循环实证);升级后退
        # 更远给更长助跑。backoff_yaw_rate 高于飞行 max_yaw_rate,
        # 边走边回正(不再原地转体,台阶边缘旋转会带滑坠落)。
        self.backoff_escalation_m=float(rospy.get_param(
            '~truth_descent_backoff_escalation_m', .10))
        self.backoff_max_distance=float(rospy.get_param(
            '~truth_descent_backoff_max_distance_m', .75))
        self.backoff_yaw_rate=float(rospy.get_param(
            '~truth_descent_backoff_yaw_rate_rps', .20))
        # 2026-08-15 卡死修复:stall 触发 backoff 前先原地回正 yaw。
        # 两次卡死实证(flight_b tyaw 漂 1.89、flight_a tyaw 漂 -1.12)
        # 表明卡死特征为 yaw 斜站台阶冻结;纯 wz 原地回正比边退边转可靠。
        self.yaw_recover_tolerance=float(rospy.get_param(
            '~truth_descent_yaw_recover_tolerance_rad', .10))
        self.yaw_recover_timeout=float(rospy.get_param(
            '~truth_descent_yaw_recover_timeout_sec', 4.0))
        self.ramp_seconds=float(rospy.get_param(
            '~truth_descent_recovery_ramp_seconds', 1.0))
        self.ramp_speed=float(rospy.get_param(
            '~truth_descent_recovery_ramp_speed_mps', .20))
        if self.backward_mode:
            # 后向下梯速度下限:backward-0.20 冒烟证实 body vx≈-0.18 时
            # stair 策略几乎不动(10s 走 0.02m),0.30 才可行(与上梯
            # flight-b backoff 同源,生产验证)。ramp 恢复同下限。
            self.descent_speed=max(self.descent_speed, .30)
            self.ramp_speed=max(self.ramp_speed, .30)
        self.fall_drop=float(rospy.get_param(
            '~truth_descent_fall_drop_m', .40))
        self.fall_window_seconds=float(rospy.get_param(
            '~truth_descent_fall_window_sec', .8))
        self.gain_up_drop=float(rospy.get_param(
            '~truth_descent_gain_up_drop_m', .55))
        self.max_center_error=float(rospy.get_param(
            '~truth_descent_max_center_error_m', .30))
        self.max_heading_error=float(rospy.get_param(
            '~truth_descent_max_heading_error_rad', .45))
        self.side_slip_min_drop=float(rospy.get_param(
            '~truth_descent_side_slip_min_drop_m', .50))
        self.side_slip_pause_seconds=float(rospy.get_param(
            '~truth_descent_side_slip_pause_sec', 2.0))
        self.side_slip_max_attempts=int(rospy.get_param(
            '~truth_descent_side_slip_max_attempts', 4))
        # 2026-08-15 v2:守卫修正窗口 —— pause 结束后给横向引导足够
        # 时间拉回再允许下一次触发(0.45m 修正需 ~10s),防连杀误杀。
        self.side_slip_guard_window_seconds=float(rospy.get_param(
            '~truth_descent_side_slip_guard_window_sec', 6.0))
        self.center_gain=float(rospy.get_param(
            '~truth_descent_flight_center_gain', .55))
        self.center_deadband=float(rospy.get_param(
            '~truth_descent_flight_center_deadband_m', .04))
        self.center_speed=float(rospy.get_param(
            '~truth_descent_flight_center_speed_mps', .12))
        self.heading_gain=float(rospy.get_param(
            '~truth_descent_flight_heading_gain', .35))
        self.max_yaw_rate=float(rospy.get_param(
            '~truth_descent_flight_max_yaw_rate_rps', .10))
        # 2026-08-15 卡死修复:目标航向加固定正偏置。
        # stair 策略下梯时 yaw 固有向正方向(+逆时针)漂 0.1~0.5 rad
        # (flight_b 漂到 1.9 / flight_a 漂到 -1.1 实测);desired 硬拉
        # 纯下行方向会让伺服大 wz 与策略步态打架 → 冻结。正偏置让
        # desired 接近策略自然姿态,伺服只做小幅修正。
        self.heading_bias=float(rospy.get_param(
            '~truth_descent_heading_bias_rad', .15))
        # 2026-08-15 v2:有界横向引导(中心偏差 → heading 修正)。
        # bias 让 body 恒有横向漂移分量(flight_b -x 漂实测 0.32m),
        # 侧滑守卫累积误杀 ALIGNMENT_LOST;附加 center 修正把 x 拉回
        # 中心线(均衡点 ≈ bias/gain ≈ 0.20m)。cap 0.20 → body 横向
        # ≤ 0.07 m/s,远低于 stair 策略冻结阈值 0.10。
        self.flight_center_steer_gain=float(rospy.get_param(
            '~truth_descent_flight_center_steer_gain', .75))
        self.flight_center_steer_max=float(rospy.get_param(
            '~truth_descent_flight_center_steer_max_rad', .20))

        # ---- 中间平台转体 ----
        # full8 2026-08-17:完成门 4.70/4.80 让转体点距 0.27m 平台台阶仅
        # 3cm,转体打滑滑下台阶 → y 安全带冻结 → TURN_TIMEOUT。默认对齐
        # 注释意图与隔离 harness 的 5.05(平台中心,28cm 滑落缓冲)。
        self.turn_start_y=float(rospy.get_param(
            '~truth_descent_turn_start_y', 5.05))
        # A long integrated run can reach the landing with a larger lateral
        # drift than the isolated baseline. Continuing the final 0.1--0.2 m
        # stride in that state drove run8 off the west edge after it had
        # already achieved the full height-drop gate. Capture that bounded
        # case into the ordinary physical landing settle/turn state; there is
        # no model reset and the turn state still has to recenter and satisfy
        # all normal landing gates.
        self.edge_drift_landing_capture_margin=float(rospy.get_param(
            '~truth_descent_edge_drift_landing_capture_margin_m', .18))
        self.edge_drift_landing_capture_count=0
        self.landing_heading_tolerance=float(rospy.get_param(
            '~truth_descent_landing_heading_tolerance_rad', .12))
        self.landing_position_tolerance=float(rospy.get_param(
            '~truth_descent_landing_position_tolerance_m', .06))
        self.landing_position_gain=float(rospy.get_param(
            '~truth_descent_landing_position_gain', .90))
        self.landing_position_deadband=float(rospy.get_param(
            '~truth_descent_landing_position_deadband_m', .05))
        self.landing_recenter_speed=float(rospy.get_param(
            '~truth_descent_landing_recenter_speed_mps', .30))
        # fixV(2026-08-19):east 平移降速 0.40→0.30。RUN1 实证 south
        # 完成后 y=2.049 已在 tread 北缘,east 0.40 平移 3s 内 x 东移进
        # tread 区(x>-3.2)z 跌 → 侧滑到东边缘外。0.30 仍在 stair 策略
        # 侧移死区(0.24)之上,但给 y 伺服拉回留出时间。
        self.landing_minimum_yaw_rate=float(rospy.get_param(
            '~truth_descent_landing_minimum_yaw_rate_rps', .24))
        # 2026-08-16:位置死区修复 —— x_err 在容差外但伺服 vx 太小
        # (≈0.057)时 plane 策略忽略,横移卡死;强制最小横向速度。
        self.landing_minimum_x_speed=float(rospy.get_param(
            '~truth_descent_landing_minimum_x_speed_mps', .30))
        # full8 2026-08-17:0.40 rps 转 180° 需 11s+,暴露在平台台阶边缘
        # 太久滑落(概率事件)。对齐上行 TURN 同平台 1.20 rps 成功参数。
        self.landing_turn_speed=float(rospy.get_param(
            '~truth_descent_landing_turn_speed_rps', 1.00))
        # 2026-08-15 TURN 掉落修复:转体步态跟随前向漂移(180° 转体
        # 漂 ~0.7m 朝 -y),机器人从平台南缘(4.52)掉进 flight_a 台阶
        # 后转体卡死。转体期间叠加世界 +y 前推补偿漂移;y 越靠平台
        # 北侧(北墙 5.52)补偿越小,避免撞墙。
        self.turn_forward_speed=float(rospy.get_param(
            '~truth_descent_turn_forward_speed_mps', .12))
        self.turn_forward_y_limit=float(rospy.get_param(
            '~truth_descent_turn_forward_y_limit_m', 5.20))
        # R31:转体时 y 滑出安全带(实测 y=4.70<4.77,脚滑出平台南缘,
        # z 掉 0.24m,转体步态死锁 4 次恢复救不回)。y<下缘时暂停转体
        # 先推回 y;backoff 同步回推 y。
        self.turn_y_safe_low=float(rospy.get_param(
            '~truth_descent_turn_y_safe_low_m', 4.77))
        # 转体 y 回推目标(安全带中心偏北,backoff 与 y 守卫共用)
        self.turn_y_keep=float(rospy.get_param(
            '~truth_descent_turn_y_keep_m', 4.85))
        # R36: 起始 yaw 与目标差 ~π(直径两端)时 atan2 error 符号在 ±π
        # 边界抖动 → wz 在 ±0.4 横跳 → 转体步态反复起步 → z 掉 0.18m
        # 滑落冻结。修复: 方向锁存(error 近 π 固定 + 方向) + 落步稳定
        # (转体前零命令等 z 稳定) + 慢启动(头 2s wz 渐进)。
        self.landing_turn_direction=None
        self.landing_settle_remaining=0.0
        self.landing_settle_seconds=float(rospy.get_param(
            '~truth_descent_landing_settle_seconds', 3.0))
        self.landing_timeout=float(rospy.get_param(
            '~truth_descent_landing_timeout_sec', 30.0))
        self.landing_recovery_max_attempts=max(0, int(rospy.get_param(
            '~truth_descent_landing_recovery_max_attempts', 2)))
        self.landing_recovery_timeout=float(rospy.get_param(
            '~truth_descent_landing_recovery_timeout_sec', 12.0))
        self.landing_recovery_minimum_yaw_rate=float(rospy.get_param(
            '~truth_descent_landing_recovery_minimum_yaw_rate_rps', .34))
        self.landing_recovery_max_position_error=float(rospy.get_param(
            '~truth_descent_landing_recovery_max_position_error_m', .40))
        self.landing_recovery_max_heading_error=float(rospy.get_param(
            '~truth_descent_landing_recovery_max_heading_error_rad', .65))
        # 2026-08-16 滑落恢复:转体步态在台阶边缘打滑时 z 骤降(台阶→
        # landing 平面落差 0.13 m),继续横移只会原地打滑(R10 实证:3 s
        # 内 z 掉 0.18 m + roll -2°,yaw 无法转动直至超时;超时 recovery
        # 因 yaw_err 超界(3.3 rad)不会触发)。检测 z 骤降 → 零命令重新
        # 落步 → world +x 回退回台阶 → 重置转体计时再试。
        self.landing_slip_drop=float(rospy.get_param(
            '~truth_descent_landing_slip_drop_m', .12))
        self.landing_slip_pause_seconds=float(rospy.get_param(
            '~truth_descent_landing_slip_pause_sec', 1.5))
        self.landing_slip_backoff_distance=float(rospy.get_param(
            '~truth_descent_landing_slip_backoff_distance_m', .40))
        self.landing_slip_backoff_speed=float(rospy.get_param(
            '~truth_descent_landing_slip_backoff_speed_mps', .30))
        self.landing_slip_max_attempts=max(0, int(rospy.get_param(
            '~truth_descent_landing_slip_max_attempts', 2)))
        # yaw 冻结检测(R13 实证:转体最后 9° 冻结 15s,z 稳定 4.21-4.25
        # 无骤降,recovery 提高 yaw rate 下限仍转不动——z_drop 检测管不到
        # 的"纯 yaw 打滑"。检测转体命令持续但 yaw 几乎不动,触发与 z 骤降
        # 同款的落步+回退恢复)
        self.landing_yaw_stall_delta_rad=float(rospy.get_param(
            '~truth_descent_landing_yaw_stall_delta_rad', .03))
        self.landing_yaw_stall_seconds=float(rospy.get_param(
            '~truth_descent_landing_yaw_stall_sec', 6.0))

        # ---- 段间转体(F2 landing 180°)----
        # 2026-08-16 重写:旧实现 position 伺服(vx/vy)+ 弱 yaw(0.30)同时
        # 进行,投影后 body=(后退,弱侧移 0.13,弱转 0.3),stair 策略转身
        # 死区(R8j 卡死实证)。新实现三阶段:align(纯侧移对中)→turn
        # (body 纯横向侧移 + 强 yaw 2.0×error)→settle(位置微调)。
        # 策略转身 = "body 横向侧移 + 同向 yaw"(TURN 实测 body=(0,0.4)
        # +wz=0.4 全程有效);R37 修复 K 后 wz 上限 1.00(对齐 TURN 提速
        # 1.00,180° 掉头 4-5s 转完减少暴露,0.40 需 11s+ 慢转打滑),
        # 下限 0.24 冲过低速率 yaw 死区。
        self.segment_turn_yaw_rate=float(rospy.get_param(
            '~truth_descent_segment_turn_yaw_rate_rps', 1.0))
        # R37: 转体命令量级对齐 TURN 成功参数 —— lateral 0.18→0.40(策略
        # 转身=body 侧移+同向 yaw,TURN 侧移 0.4 有效、0.18 弱侧移打滑;
        # full14 实证:baseline launch 默认 0.18 覆盖代码默认,实际侧移
        # 仍是 R37 失败量级 → yaw 转 104° 后冻结);drift_limit 0.40→0.60
        # (正常转体的侧移投影东漂 ~0.3m,护栏误触打断转体,R37 每次 turn
        # 窗口 0.4-1s 反复打断 → 步态无法进入稳定循环 → 打滑冻结)。
        self.segment_turn_lateral_speed=float(rospy.get_param(
            '~truth_descent_segment_turn_lateral_speed_mps', .40))
        self.segment_turn_side_speed=float(rospy.get_param(
            '~truth_descent_segment_turn_side_speed_mps', .40))
        self.segment_turn_drift_limit=float(rospy.get_param(
            '~truth_descent_segment_turn_drift_limit_m', .60))
        self.segment_turn_y_keep=float(rospy.get_param(
            '~truth_descent_segment_turn_y_keep_m', 2.05))
        # 掉头 x 安全中心(landing 西半平坦区),NOT flight_b 中心(-2.485):
        # R24 实证 x=-3.30 已是 flight_b tread 西缘(z 骤降),R20/R21
        # 掉头东滑/侧翻同源;掉头必须在 x≈-3.7 完成,完成后 FLIGHT_B 段
        # 再自行东进阶梯。y_keep=2.05(平台北缘,平台 y∈[1.05,2.05]):
        # full17 实证掉头起步 y=2.22 已在 flight_a step_0 上(y>2.05),
        # 反转掉头(172°)南漂回平台内;y_low=1.7 防南漂出平台。
        self.segment_turn_x_keep=float(rospy.get_param(
            '~truth_descent_segment_turn_x_keep_m', -3.7))
        # y 安全区间(landing 平坦带):y_keep±0.35。R27 实证 y 伺服把 y 从
        # 2.17 推到 2.24 后步态打滑推不动(冻结 37.5s,恢复 4 次无效,
        # SEGMENT_TURN_TIMEOUT)——y 只需在平坦带内即可掉头,不必收敛到
        # 精确 2.4;区间外才拉回,区间内零命令(y_keep 伺服打滑会卡死)。
        # full12 实证:掉头完成时 y 南漂到 2.072,±0.3 带(y_low 2.1)判
        # 失败 → settle 死循环超时;放宽下界到 2.05(flight_b tread 边缘
        # 再低 2cm,z 掉 0.45m,slip 检测 z_drop>0.12 兜底)。
        self.segment_turn_y_band=float(rospy.get_param(
            '~truth_descent_segment_turn_y_band_m', 0.35))
        # run32 reached a verified flat slab at y=1.62, z=2.917 and
        # pitch=-1 deg.  The old symmetric lower edge (1.70) kept the stair
        # policy in weak translational alignment and never issued the strong
        # lateral+yaw turn.  The slab extends south to about y=1.05, so 1.45
        # retains edge margin while admitting this valid physical landing.
        self.segment_turn_y_low=float(rospy.get_param(
            '~truth_descent_segment_turn_y_low_m', 1.45))
        self.segment_turn_y_high=self.segment_turn_y_keep+\
            self.segment_turn_y_band
        self.segment_turn_position_gain=float(rospy.get_param(
            '~truth_descent_segment_turn_position_gain', .60))
        self.segment_turn_position_tolerance=float(rospy.get_param(
            '~truth_descent_segment_turn_position_tolerance_m', .12))
        self.segment_turn_heading_tolerance=float(rospy.get_param(
            '~truth_descent_segment_turn_heading_tolerance_rad', .12))
        self.segment_turn_timeout=float(rospy.get_param(
            '~truth_descent_segment_turn_timeout_sec', 45.0))
        # fullflow run20: the first flight physically reached the flat F2
        # landing, but retained ~19 deg pitch.  The stair gait then produced
        # no yaw after both ordinary re-plant attempts.  A bounded FixedStand
        # transaction is safer than terminating the round trip (and unlike a
        # model-state correction it remains a physical recovery).
        self.segment_turn_fixed_stand_enabled=bool(rospy.get_param(
            '~truth_descent_segment_turn_fixed_stand_enabled', True))
        self.segment_turn_fixed_stand_seconds=float(rospy.get_param(
            '~truth_descent_segment_turn_fixed_stand_sec', 3.5))
        self.segment_turn_fixed_stand_max_attempts=max(0, int(
            rospy.get_param(
                '~truth_descent_segment_turn_fixed_stand_max_attempts', 1)))
        # 段 2 起步东移对齐目标(F2 landing 平台内 y=2.0,x 对齐
        # flight_b_floor_0 中心线 -2.485;平台 y∈[1.05,2.05],北缘
        # y>2.05 已是楼梯)。
        self.east_align_y_target=float(rospy.get_param(
            '~truth_descent_east_align_y_m', 1.85))
        self.east_align_minimum_speed=float(rospy.get_param(
            '~truth_descent_east_align_minimum_speed_mps', .30))
        # fixV(2026-08-19):south 目标 2.0→1.85。RUN1 实证 south 完成
        # y=2.049 已贴 tread 北缘(y>2.0),east 平移 3s 即踩进 tread 侧滑。
        # 1.85 在平台内(y∈[1.05,2.05])多留 0.2m,抵掉 east 平移的 y
        # 北漂耦合(~0.1m/m × 东移 1.1m ≈ 0.11)后仍 ≥2.0。
        # fixU(2026-08-19):EAST_ALIGN 侧移中 z 跌到该阈值以下 = 已踩进
        # flight_b tread(平台 z≈2.96-3.09,tread 底 z≈2.60)。stair 策略
        # 在台阶上只接受纯 body 轴命令,世界 vx 平移在 tread 边缘侧滑
        # (RUN2 实证 z 3.09→2.60 滑落后 vy 南拉冻结 30s);立即切
        # FLIGHT_B,由纯轴下梯 + yaw 对中 + stall recovery 兜底接管。
        self.east_align_tread_z=float(rospy.get_param(
            '~truth_descent_east_align_tread_z_m', 2.75))
        # fixV3:fixU z 交棒前 x 距 flight_b 中心的容差 —— 到位才交棒,
        # 防止 tread 西缘起步(段2 run3 实证 0.62m 偏位卡死)。
        self.east_align_tread_x_tol=float(rospy.get_param(
            '~truth_descent_east_align_tread_x_tol_m', 0.35))

        # ---- 完成门与阶段运行状态 ----
        self.flight_b_descent_drop=float(rospy.get_param(
            '~truth_descent_flight_b_drop_gate_m', 1.05))
        self.flight_a_descent_drop=float(rospy.get_param(
            '~truth_descent_flight_a_drop_gate_m', 2.45))
        self.flight_a_end_y=float(rospy.get_param(
            '~truth_descent_flight_a_end_y', 2.25))
        self.descent_start_z=None
        self.flight_started_at=None
        self.flight_stall_since=None
        self.flight_stall_window_start_z=None
        self.flight_stall_recoveries=0
        self.flight_max_stall_recoveries=int(rospy.get_param(
            '~truth_descent_flight_max_stall_recoveries', 2))
        self.flight_pause_until=None
        self.flight_ramp_until=None
        self.flight_backoff_until=None
        self.flight_yaw_recover_until=None
        self.flight_descent_min_z=None
        self.fall_window_until=None
        self.fall_window_start_z=None
        self.side_slip_attempts=0
        self.side_slip_pause_until=None
        self.side_slip_next_check=None
        self.landing_started_at=None
        self.landing_recovery_active=False
        self.landing_recovery_attempts=0
        self.landing_heading_error=None
        self.landing_position_error=None
        self.landing_z0=None
        self.landing_slip_active=False
        self.landing_slip_until=None
        self.landing_slip_backoff_until=None
        self.landing_slip_recoveries=0
        self.landing_yaw_stall_accum=0.0
        self.landing_last_yaw=None
        self.landing_last_yaw_at=None
        self.segment_turn_started_at=None
        self.segment_turn_stage='align'
        # R36: 段间转体同样存在起始 yaw 与目标差 ~π 的 ±π 边界抖动问题
        # (FLIGHT_A 后 yaw≈-π/2,目标 flight_b heading≈+π/2,差≈π),同款
        # 方向锁存。
        self.segment_turn_direction=None
        # R38: SEGMENT_TURN 完成标志 —— self.segment 段 1 完成后仍为 1
        # (2047 行赋值),不能区分段 1/段 2;EAST_ALIGN 只用于段 2。
        self.segment_turn_done=False
        # R37: turn 阶段慢启动计时(align→turn 切换时置位)
        self.segment_turn_turn_started_at=None
        self.segment_turn_fixed_stand_attempts=0
        self.segment_turn_fixed_stand_started_at=None
        self.segment_turn_fixed_stand_last_request=None
        self.segment_policy_loading_started_at=None
        self.segment_policy_loading_last_request=None
        self.segment_policy_loading_request_count=0
        self.segment_policy_loading_retry_period=max(0.5, float(
            rospy.get_param(
                '~truth_descent_segment_policy_retry_period_sec', 2.0)))
        self.segment_policy_loading_timeout=max(2.0, float(rospy.get_param(
            '~truth_descent_segment_policy_loading_timeout_sec', 12.0)))
        self.segment_policy_loading_max_requests=max(1, int(rospy.get_param(
            '~truth_descent_segment_policy_loading_max_requests', 4)))
        # The entry guide can be preceded by a plane-policy load (notably the
        # bounded F2 failure-return path).  Keep the subsequent stair-policy
        # transaction independent: a stale generic ``policy_requested`` flag
        # must never suppress the new request, and a lost acknowledgement
        # must not leave this required node waiting forever.
        self.stair_policy_loading_started_at=None
        self.stair_policy_loading_last_request=None
        self.stair_policy_loading_request_count=0
        self.stair_policy_loading_retry_period=max(0.5, float(
            rospy.get_param(
                '~truth_descent_stair_policy_retry_period_sec', 2.0)))
        self.stair_policy_loading_timeout=max(2.0, float(rospy.get_param(
            '~truth_descent_stair_policy_loading_timeout_sec', 12.0)))
        self.stair_policy_loading_max_requests=max(1, int(rospy.get_param(
            '~truth_descent_stair_policy_loading_max_requests', 4)))
        self.segment_policy_lower_floor_z=max(0.6, float(rospy.get_param(
            '~truth_descent_segment_policy_lower_floor_z_m', 1.20)))
        self.segment=0
        self.policy_warmup_seconds=float(rospy.get_param(
            '~policy_warmup_seconds', 1.2))
        self.policy_warmup_until=None
        self.hold_rl_during_ascent=bool(rospy.get_param(
            '~hold_rl_during_ascent', True))

        os.makedirs(os.path.join(self.out, 'logs'), exist_ok=True)
        if self.offline_layout and os.path.isfile(self.offline_layout):
            try:
                shutil.copy2(self.offline_layout,
                             os.path.join(self.out, 'layout_metadata.json'))
            except OSError:
                pass
        self._load_landing_center()

        # ---- 发布/订阅 ----
        self._pause_pub=rospy.Publisher(
            rospy.get_param('~goal_executor_pause_topic',
                            '/simenv/goal_executor_pause'),
            Bool, queue_size=1, latch=True)
        self.pub=rospy.Publisher('/simenv/rl_policy_request', String,
                                 queue_size=1, latch=True)
        self.cmd=rospy.Publisher(rospy.get_param('~command_topic', '/cmd_vel'),
                                 Twist, queue_size=2)
        self.joy=rospy.Publisher('/joy', Joy, queue_size=2)
        self.state=rospy.Publisher(self.state_topic, String,
                                   queue_size=1, latch=True)
        self.first_floor_state_pub=rospy.Publisher(
            rospy.get_param('~first_floor_state_topic',
                            '/simenv/first_floor_state'),
            String, queue_size=1, latch=True)
        if self.mission_trigger_topic:
            rospy.Subscriber(self.mission_trigger_topic, String,
                             self.on_mission_trigger, queue_size=5)
        self.odom_topic=rospy.get_param('~odom_topic', '/Odometry')
        rospy.Subscriber(self.odom_topic, Odometry, self.on_odom, queue_size=10)
        rospy.Subscriber('/gazebo/model_states', ModelStates,
                         self.on_truth_states, queue_size=2)
        rospy.Subscriber('/gazebo/link_states', LinkStates,
                         self.on_truth_links, queue_size=1)
        rospy.Subscriber('/rl_takeover_status', String,
                         self.on_policy_status, queue_size=4)
        rospy.Subscriber('/locomotion_ready', Bool,
                         self.on_locomotion_ready, queue_size=2)
        rospy.Subscriber('/fixed_stand_ready', Bool,
                         self.on_fixed_stand_ready, queue_size=2)
        rospy.Subscriber('/a1_gazebo/joint_states', JointState,
                         self.on_joint_states, queue_size=1)
        rospy.Timer(rospy.Duration(.02), self.tick)
        rospy.on_shutdown(self.save)

    # ------------------------------------------------------------------ #
    # 回调
    # ------------------------------------------------------------------ #
    def _begin_second_floor_failure_return(self, token):
        """Recover an upstream F2/F3 failure through F2->F1 and home.

        This is locomotion continuity only.  It never publishes an F3-ready
        token and never changes any floor's room, two-view, or danger credit.
        The current robot may be beside the F2->F3 staircase rather than at
        the F2->F1 descent lip, so reload the plane policy and use the bounded
        truth entry guide before starting the ordinary physical descent.
        """
        self.started=time.monotonic()
        self.trigger_wall_time=time.time()
        self.trigger_ros_time=rospy.Time.now().to_sec()
        self.start_floor_number=2
        self.end_floor_number=1
        self.total_segments=1
        self.segment=0
        self.flight_floor_index=0
        self._geometry_dirty=True
        self.truth_entry_guide=True
        self.skip_entry_guide=False
        self.truth_landing_center=None
        self._load_landing_center()
        self.lower_floor_failure_return_mode='F2_TO_F1_AFTER_{}'.format(token)
        self.active_policy=self.plane_policy
        self.policy_loaded=False
        self.policy_requested=True
        self.lower_floor_policy_loading_started=time.monotonic()
        self.lower_floor_policy_last_request=None
        self.lower_floor_policy_request_count=0
        self.phase='SECOND_FLOOR_FAILURE_RETURN_POLICY_LOADING'
        self.state.publish(String(data=self.phase))
        self.cmd.publish(Twist())
        rospy.logwarn(
            'Upstream mission failed on F2; preserving strict failure and '
            'starting bounded F2->F1->home return (token=%s).', token)

    def _second_floor_failure_policy_tick(self):
        now=time.monotonic()
        if self.policy_loaded:
            self._begin_segment()
            self.state.publish(String(data=self.phase))
            rospy.loginfo(
                'Plane policy ready; guiding failed upstream mission from '
                'F2 to the physical descent entrance.')
            self._record_trace(force=True)
            return
        if (self.lower_floor_policy_last_request is None or
                (now-self.lower_floor_policy_last_request >=
                 self.lower_floor_policy_retry_period and
                 self.lower_floor_policy_request_count <
                 self.lower_floor_policy_max_requests)):
            self.pub.publish(String(data=self.plane_policy))
            self.lower_floor_policy_last_request=now
            self.lower_floor_policy_request_count+=1
        self.cmd.publish(Twist())
        self.hold_rl()
        if (self.lower_floor_policy_loading_started is not None and
                now-self.lower_floor_policy_loading_started >=
                self.lower_floor_policy_timeout):
            self._terminal(
                'SECOND_FLOOR_FAILURE_RETURN_POLICY_TIMEOUT',
                'second_floor_failure_return_policy_timeout', error=True)
            return
        self._record_trace()

    def on_mission_trigger(self, message):
        token=str(message.data).strip()
        if self.phase == 'WAIT_F3' and token in self.mission_failure_tokens:
            if getattr(self, 'terminate_on_mission_failure', False):
                self._terminal(token, token.lower(), error=True)
                return
            # F3 publishes EXPLORATION_FAILED before its bounded truth return
            # has reached the stair wait zone.  A later
            # THIRD_FLOOR_STAIR_RETURN_PARTIAL_READY is still a valid physical
            # trigger for the trip home (the run remains INCOMPLETE).  Exiting
            # this required node here used to tear down roslaunch while that
            # return was in progress.  Latch the failed exploration evidence
            # and keep waiting under the existing finite trigger watchdog.
            # If the upstream chain failed before ever reaching F3, there can
            # be no later bounded F3 stair-return token.  physdedupe_lipfix
            # failed on F2 at z=2.91 m and this required worker consequently
            # waited under its 2400 s trigger watchdog.  Terminate explicitly
            # in that physically unambiguous case; only a robot already at F3
            # height may wait for the bounded partial return.
            if self._dispatch_failure_return_from_truth(token):
                return
            if self._dispatch_third_floor_failure_return_at_landing(token):
                return
            self.pending_mission_failure_token=token
            self.pending_mission_failure_wall_time=time.time()
            self.state.publish(String(
                data='WAITING_FOR_BOUNDED_THIRD_FLOOR_STAIR_RETURN'))
            rospy.logwarn(
                'F3 exploration reported %s; waiting for bounded stair '
                'return token (strict completion remains false).', token)
            return
        if self.phase == 'WAIT_F3' and token in self.mission_trigger_tokens:
            self.started=time.monotonic()
            self.trigger_wall_time=time.time()
            self.trigger_ros_time=rospy.Time.now().to_sec()
            self.segment_started=self.started
            self._begin_segment()
            self.state.publish(String(data=self.phase))

    def _dispatch_failure_return_from_truth(self, token):
        """Route a failed mission home once its physical floor is known."""
        truth_pose=getattr(self, 'truth_pose', None)
        if truth_pose is None:
            return False
        minimum_height=float(getattr(
            self, 'minimum_third_floor_failure_wait_height', 4.80))
        z=float(truth_pose[2])
        if z >= minimum_height:
            return False
        if z <= float(getattr(
                self, 'first_floor_failure_return_maximum_z', 1.20)):
            self.started=time.monotonic()
            self.trigger_wall_time=time.time()
            self.trigger_ros_time=rospy.Time.now().to_sec()
            self.lower_floor_failure_return_mode=(
                'F1_HOME_AFTER_{}'.format(token))
            rospy.logwarn(
                'Upstream mission failed on F1 (z=%.3f); returning '
                'directly to the original start without completion credit.',
                z)
            if self._home_pose_upright():
                self._begin_home_return()
            else:
                self._begin_home_fall_recovery_wait(
                    'failed_f1_exploration_trigger')
        elif (float(getattr(
                self, 'second_floor_failure_return_minimum_z', 2.30))
              <= z <= float(getattr(
                self, 'second_floor_failure_return_maximum_z', 3.40))):
            self._begin_second_floor_failure_return(token)
        else:
            rospy.logerr(
                'Upstream exploration failed at an unsafe inter-floor '
                'height (z=%.3f); bounded return cannot be staged.', z)
            self._terminal(
                'STAIR_DESCENT_NOT_TRIGGERED_SOURCE_FLOOR_FAILED',
                'stair_descent_not_triggered_source_floor_failed', error=True)
        self.pending_mission_failure_token=None
        self.pending_mission_failure_wall_time=None
        return True

    def _dispatch_third_floor_failure_return_at_landing(self, token):
        """Start descent when failed exploration physically reached F3 lip.

        Room quality remains failed.  This gate proves only that the robot is
        upright-height on the generated upper landing and close enough for
        the normal bounded descent entry guide to take over.  It prevents a
        rejected intermediate shaping waypoint from suppressing the entire
        trip home, as happened in run63 at (-0.92, 1.74).
        """
        pose=getattr(self, 'truth_pose', None)
        landing=getattr(self, 'truth_landing_center', None)
        if pose is None or landing is None:
            return False
        if float(pose[2]) < float(getattr(
                self, 'minimum_third_floor_failure_wait_height', 4.80)):
            return False
        distance=math.hypot(float(pose[0])-float(landing[0]),
                            float(pose[1])-float(landing[1]))
        if distance > float(getattr(
                self, 'third_floor_failure_landing_trigger_radius', 1.25)):
            return False
        self.started=time.monotonic()
        self.trigger_wall_time=time.time()
        self.trigger_ros_time=rospy.Time.now().to_sec()
        self.pending_mission_failure_token=None
        self.pending_mission_failure_wall_time=None
        self._begin_segment()
        self.state.publish(String(data=self.phase))
        rospy.logwarn(
            'F3 exploration failed but physically reached the descent '
            'landing (distance=%.3f m); starting bounded trip home without '
            'granting room-completion credit (token=%s).', distance, token)
        self._record_trace(force=True)
        return True

    def _poll_persisted_mission_failure(self):
        """Recover a short-lived latched failure publisher from disk evidence."""
        if not getattr(self, 'out', None):
            return
        now=time.time()
        if now-getattr(self, 'persisted_failure_poll_wall_time', 0.0) < 1.0:
            return
        self.persisted_failure_poll_wall_time=now
        handoff_path=os.path.join(self.out, 'third_floor_handoff.json')
        try:
            with open(handoff_path, 'r', encoding='utf-8') as stream:
                handoff=json.load(stream)
        except (OSError, ValueError, TypeError):
            return
        if str(handoff.get('state', '')).strip() in self.mission_failure_tokens:
            rospy.logwarn(
                'Recovered missed mission failure trigger from %s.',
                handoff_path)
            self.on_mission_trigger(String(data=str(handoff['state'])))

    def on_policy_status(self, message):
        if ('policy_reloaded:' in message.data and
                os.path.basename(self.active_policy) in message.data):
            # Policy acknowledgements are global.  During F1->F2/F2->F3 the
            # ascent manager loads the same stair checkpoint, so this waiting
            # descent node receives that acknowledgement too.  Publishing
            # STAIR_DESCENT_POLICY_READY from WAIT_F3 falsely claims cmd_vel
            # ownership and makes the planar goal executor yield throughout
            # upper-floor exploration.  Only an acknowledgement requested by
            # this node's active loading phase may change its state.
            if self.phase not in ('STAIR_DESCENT_POLICY_LOADING',
                                  'STAIR_DESCENT_POLICY_QUEUED_STAND',
                                  'STAIR_DESCENT_POLICY_RL_REENTRY',
                                  'STAIR_DESCENT_SEGMENT_POLICY_LOADING',
                                  'SECOND_FLOOR_FAILURE_RETURN_POLICY_LOADING',
                                  'FIRST_FLOOR_HOME_POLICY_LOADING'):
                return
            self.policy_loaded=True
            ready=('FIRST_FLOOR_HOME_POLICY_READY'
                   if self.phase == 'FIRST_FLOOR_HOME_POLICY_LOADING'
                   else 'SECOND_FLOOR_FAILURE_RETURN_POLICY_READY'
                   if self.phase ==
                   'SECOND_FLOOR_FAILURE_RETURN_POLICY_LOADING'
                   else 'STAIR_DESCENT_POLICY_READY')
            if ready == 'FIRST_FLOOR_HOME_POLICY_READY':
                self._publish_milestone(ready)
            else:
                self.state.publish(String(data=ready))
        elif 'policy_reload_failed:' in message.data and \
                self.phase in ('STAIR_DESCENT_POLICY_LOADING',
                               'STAIR_DESCENT_POLICY_QUEUED_STAND',
                               'STAIR_DESCENT_POLICY_RL_REENTRY',
                               'STAIR_DESCENT_SEGMENT_POLICY_LOADING',
                               'SECOND_FLOOR_FAILURE_RETURN_POLICY_LOADING',
                               'FIRST_FLOOR_HOME_POLICY_LOADING'):
            phase=('FIRST_FLOOR_HOME_POLICY_FAILED'
                   if self.phase == 'FIRST_FLOOR_HOME_POLICY_LOADING'
                   else 'STAIR_DESCENT_POLICY_FAILED')
            self._terminal(phase, phase.lower(), error=True)

    def on_locomotion_ready(self, message):
        self.locomotion_ready=bool(message.data)
        if self.locomotion_ready:
            self.locomotion_ready_at=time.monotonic()

    def on_fixed_stand_ready(self, message):
        now=time.monotonic()
        self.fixed_stand_ready=bool(message.data)
        if self.fixed_stand_ready:
            self.fixed_stand_ready_at=now
        else:
            self.fixed_stand_not_ready_at=now
            self.fixed_stand_ready_at=None

    def on_joint_states(self, message):
        velocities=list(getattr(message, 'velocity', ()) or ())
        velocity_rms=(math.sqrt(sum(value*value for value in velocities) /
                                len(velocities)) if velocities else None)
        received=time.monotonic()
        with self._joint_lock:
            self.joint_names=list(message.name)
            self.joint_positions=list(message.position)
            self.joint_velocity_rms=velocity_rms
            self.joint_received_at=received
            if len(message.name) == len(message.position):
                self.joint_samples.append((
                    received, dict(zip(message.name, message.position)),
                    velocity_rms))

    def on_odom(self, message):
        self.odom_seen=True
        p=message.pose.pose.position
        self.pose=(p.x, p.y, p.z, yaw(message.pose.pose.orientation))

    def on_truth_states(self, message):
        try:
            index=message.name.index('a1_gazebo')
            pose=message.pose[index]
            q=pose.orientation
            roll, pitch, _ = euler_from_quaternion(
                (q.x, q.y, q.z, q.w))
            self.truth_pose=(float(pose.position.x), float(pose.position.y),
                             float(pose.position.z), yaw(pose.orientation))
            self.truth_attitude=(roll, pitch)
            if index < len(message.twist):
                twist=message.twist[index]
                self.truth_twist=(float(twist.linear.x),
                                  float(twist.linear.y),
                                  float(twist.linear.z),
                                  float(twist.angular.z))
        except (ValueError, IndexError):
            pass

    def on_truth_links(self, message):
        # LinkStates 是静态几何;仅在段切换(flight_floor_index 变化)后重新解析。
        if not self._geometry_dirty and self.truth_step_pose is not None:
            return
        floor_suffix=str(self.flight_floor_index)
        def find(*suffixes):
            for i, name in enumerate(message.name):
                if any(name.endswith(suffix) for suffix in suffixes):
                    pose=message.pose[i]
                    return (float(pose.position.x), float(pose.position.y),
                            float(pose.position.z), yaw(pose.orientation))
            return None
        def cardinal_heading(dx, dy):
            if abs(dx) >= abs(dy):
                return 0.0 if dx >= 0.0 else math.pi
            return math.pi/2.0 if dy >= 0.0 else -math.pi/2.0
        a0=find('::stair_flight_a_floor_{}_step_0'.format(floor_suffix))
        a1=find('::stair_flight_a_floor_{}_step_1'.format(floor_suffix))
        a9=find('::stair_flight_a_floor_{}_step_9'.format(floor_suffix))
        b0=find('::stair_flight_b_floor_{}_step_0'.format(floor_suffix))
        b1=find('::stair_flight_b_floor_{}_step_1'.format(floor_suffix))
        b9=find('::stair_flight_b_floor_{}_step_9'.format(floor_suffix))
        if a0 is None or a1 is None or a9 is None or \
                b0 is None or b1 is None or b9 is None:
            return
        self.truth_step_pose=a0
        self.truth_step_next_pose=(a1[0], a1[1])
        self.truth_flight_a_heading=cardinal_heading(
            a1[0]-a0[0], a1[1]-a0[1])
        self.truth_flight_a_top=a9
        self.truth_flight_b_pose=b0
        self.truth_flight_b_next_pose=(b1[0], b1[1])
        self.truth_flight_b_heading=cardinal_heading(
            b1[0]-b0[0], b1[1]-b0[1])
        self.truth_flight_b_top=b9
        self._geometry_dirty=False

    # ------------------------------------------------------------------ #
    # 工具
    # ------------------------------------------------------------------ #
    def hold_rl(self):
        if not self.hold_rl_during_ascent:
            return
        message=Joy()
        message.header.stamp=rospy.Time.now()
        message.axes=[0.0]*8
        message.buttons=[0]*12
        message.buttons[3]=1  # Unitree RL /cmd_vel 模式
        self.joy.publish(message)

    def _publish_fixed_stand(self):
        message=Joy()
        message.header.stamp=rospy.Time.now()
        message.axes=[0.0]*8
        message.buttons=[0]*12
        message.buttons[1]=1
        self.joy.publish(message)

    _FSM_JOINT_ORDER=(
        'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
        'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
        'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint',
        'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint')

    @classmethod
    def _symmetric_support_target(cls, raw_mean):
        """Project a gait-window mean to a symmetric load-bearing stance."""
        if len(raw_mean) != 12 or not all(math.isfinite(v) for v in raw_mean):
            return None
        # All four legs use the same production FixedStand convention. Keep
        # the observed mean body height, while removing a captured trot phase.
        thigh=sum(raw_mean[index] for index in (1, 4, 7, 10))/4.0
        calf=sum(raw_mean[index] for index in (2, 5, 8, 11))/4.0
        thigh=min(1.15, max(.65, thigh))
        calf=min(-1.50, max(-2.10, calf))
        target=[]
        for _ in range(4):
            target.extend((0.0, thigh, calf))
        return target

    def _capture_stand_target(self):
        now=time.monotonic()
        with self._joint_lock:
            names=list(self.joint_names) if self.joint_names else None
            positions=(list(self.joint_positions)
                       if self.joint_positions is not None else None)
            rms=self.joint_velocity_rms
            received=self.joint_received_at
            samples=list(self.joint_samples)
        if (not names or positions is None or received is None or
                now-received > 2.0 or
                (rms is not None and
                 rms > self.policy_switch_joint_velocity_rms)):
            return False
        window_start=max(
            now-self.policy_switch_snapshot_window,
            self.policy_switch_snapshot_started
            if self.policy_switch_snapshot_started is not None else -math.inf)
        usable=[]
        for stamp, by_name, sample_rms in samples:
            if stamp < window_start:
                continue
            try:
                usable.append((stamp, [float(by_name[name])
                                       for name in self._FSM_JOINT_ORDER],
                               sample_rms))
            except (KeyError, TypeError, ValueError):
                continue
        if len(usable) < self.policy_switch_snapshot_min_samples or \
                usable[-1][0]-usable[0][0] < \
                .75*self.policy_switch_snapshot_window:
            return False
        raw_mean=[sum(sample[1][index] for sample in usable)/len(usable)
                  for index in range(12)]
        target=self._symmetric_support_target(raw_mean)
        if target is None:
            return False
        pair_spread=max(
            abs(raw_mean[left]-raw_mean[right])
            for left, right in ((0, 3), (1, 4), (2, 5),
                                (6, 9), (7, 10), (8, 11)))
        self.stand_target_evidence={
            'source':'stable_window_symmetric_support',
            'sample_count':len(usable),
            'window_span_sec':round(usable[-1][0]-usable[0][0], 4),
            'capture_joint_velocity_rms':(
                round(rms, 5) if rms is not None else None),
            'raw_pair_spread_max_rad':round(pair_spread, 5),
            'raw_mean':[round(value, 6) for value in raw_mean],
            'target':[round(value, 6) for value in target],
        }
        try:
            rospy.set_param(self._stand_target_param, target)
        except Exception as error:
            rospy.logerr('Descent could not set FixedStand target: %s', error)
            return False
        self._stand_target_applied=True
        rospy.loginfo(
            'Descent captured symmetric support target from %d samples '
            '(span=%.3fs RMS=%s raw_pair_spread=%.3frad target=%s).',
            len(usable), usable[-1][0]-usable[0][0],
            'unknown' if rms is None else '%.3f' % rms, pair_spread,
            ','.join('%.3f' % value for value in target))
        return True

    def _clear_parameter_profile(self, names):
        ok=True
        for name in names:
            try:
                if rospy.has_param(name):
                    rospy.delete_param(name)
            except (KeyError, rospy.ROSException):
                pass
            except Exception as error:
                ok=False
                rospy.logerr('Cannot clear policy-boundary param %s: %s',
                             name, error)
        return ok

    def _clear_stand_target(self):
        if not self._stand_target_applied:
            return True
        ok=self._clear_parameter_profile((self._stand_target_param,))
        if ok:
            self._stand_target_applied=False
        return ok

    def _arm_fast_takeover(self):
        self._fast_takeover_profile_owned=True
        try:
            for name, value in self._fast_takeover_params.items():
                rospy.set_param(name, value)
        except Exception as error:
            rospy.logerr('Cannot arm stair takeover profile: %s', error)
            self._clear_fast_takeover()
            return False
        return True

    def _clear_fast_takeover(self):
        if not self._fast_takeover_profile_owned:
            return True
        ok=self._clear_parameter_profile(self._fast_takeover_params)
        if ok:
            self._fast_takeover_profile_owned=False
        return ok

    def _arm_fast_fixed_stand(self):
        if not self.policy_switch_fast_fixed_stand:
            return True
        self._fast_fixed_stand_profile_owned=True
        try:
            for name, value in self._fast_fixed_stand_params.items():
                rospy.set_param(name, value)
        except Exception as error:
            rospy.logerr('Cannot arm FixedStand profile: %s', error)
            self._clear_fast_fixed_stand()
            return False
        return True

    def _clear_fast_fixed_stand(self):
        if not self._fast_fixed_stand_profile_owned:
            return True
        ok=self._clear_parameter_profile(self._fast_fixed_stand_params)
        if ok:
            self._fast_fixed_stand_profile_owned=False
        return ok

    def _policy_switch_body_stable(self):
        if self.truth_twist is None:
            return True
        vx, vy, vz, wz=self.truth_twist
        return (math.hypot(vx, vy) <= .18 and abs(vz) <= .12 and
                abs(wz) <= .25)

    def _publish_truth_world_command(self, vx, vy, wz=0.0):
        """世界系命令,经 Gazebo 躯干 yaw 投影到 body 系(与上梯同源)。"""
        if self.truth_pose is None:
            return
        robot_yaw=self.truth_pose[3]
        command=Twist()
        command.linear.x=vx*math.cos(robot_yaw)+vy*math.sin(robot_yaw)
        command.linear.y=-vx*math.sin(robot_yaw)+vy*math.cos(robot_yaw)
        command.angular.z=wz
        rospy.loginfo_throttle(
            1.0,
            'DSTPUB phase=%s world=(%.3f,%.3f,%.3f) body=(%.3f,%.3f,%.3f) '
            'truth_z=%.3f drop=%.3f pitch=%.1fdeg seg=%d',
            self.phase, vx, vy, wz, command.linear.x, command.linear.y,
            command.angular.z,
            self.truth_pose[2] if self.truth_pose is not None else float('nan'),
            self._drop(),
            math.degrees(self.truth_attitude[1]) if self.truth_attitude
            else float('nan'), self.segment)
        self.cmd.publish(command)
        self.hold_rl()

    def _drop(self):
        if self.descent_start_z is None or self.truth_pose is None:
            return 0.0
        # 正值 = 已下降(段起点 z 减去当前 z)
        return self.descent_start_z-self.truth_pose[2]

    def _flight_y_advance(self):
        """沿下梯方向的 y 累计前进量(flight-B 沿 +y,flight-A 沿 -y)。

        fixV4 新增:stall 武装盲区修复 —— 段2 FLIGHT_B 起步即卡时
        drop≈0,用 y 进展判定是否已进入楼梯区域。"""
        if self.flight_start_y is None or self.truth_pose is None:
            return 0.0
        return abs(self.truth_pose[1]-self.flight_start_y)

    def _flight_y_advance_from(self, y0):
        """相对给定 y 起点的前进量(绝对值),供 stall 窗口冻结判定。"""
        # Phase/reset callbacks run on separate rospy threads and may clear a
        # stall window between the caller's armed check and this calculation.
        # Treat that transition as a fresh window, never as a timer-thread
        # exception that silently kills the entire descent state machine.
        if self.truth_pose is None or y0 is None:
            return 0.0
        return abs(self.truth_pose[1]-y0)

    def _elapsed(self):
        if self.started is None:
            return 0.0
        return time.monotonic()-self.started

    def _segment_elapsed(self):
        if self.segment_started is None:
            return 0.0
        return time.monotonic()-self.segment_started

    def _load_landing_center(self):
        """从 model.sdf 解析起点 landing 中心(所有 landing x,y 相同)。"""
        if not self.offline_stair_model or not os.path.isfile(
                self.offline_stair_model):
            return
        landing_name='stair_floor_landing_floor_{}'.format(
            self.start_floor_number-1)
        try:
            root=ET.parse(self.offline_stair_model).getroot()
            link=next(item for item in root.iter('link')
                      if item.get('name') == landing_name)
            values=(link.findtext('pose') or '').split()
            if len(values) < 2:
                return
            self.truth_landing_center=(float(values[0]), float(values[1]))
        except (OSError, StopIteration, ValueError, ET.ParseError):
            rospy.logwarn('Cannot parse landing pose from %s',
                          self.offline_stair_model)

    def _flight_b_center_x(self):
        if self.truth_flight_b_pose is not None:
            return self.truth_flight_b_pose[0]
        return -2.485

    def _flight_a_center_x(self):
        if self.truth_step_pose is not None:
            return self.truth_step_pose[0]
        return -4.015

    def _descent_heading(self, up_heading):
        """上行方向的相反方向 = 下行方向(wrap 到 [-pi, pi])。"""
        result=up_heading+math.pi
        return math.atan2(math.sin(result), math.cos(result))

    def _flight_heading(self, up_heading):
        """本段 flight 的下梯朝向:正向 = 下行方向;后向 = 上行方向。"""
        if not self.backward_mode:
            return self._descent_heading(up_heading)
        return up_heading

    def _record_trace(self, force=False):
        now=time.monotonic()
        if not force and now-self.last_log <= .1:
            return
        pose=(self.truth_pose if self.truth_entry_guide and
              self.truth_pose is not None else self.pose)
        if pose is None:
            return
        item={'t': round(self._elapsed(), 3),
              'ros_time': round(rospy.Time.now().to_sec(), 3),
              'wall_time': round(time.time(), 3),
              'phase': self.phase,
              'x': pose[0], 'y': pose[1], 'z': pose[2], 'yaw': pose[3],
              'segment': self.segment,
              'descent_drop': round(self._drop(), 3),
              'stall_recoveries': self.flight_stall_recoveries,
              'fall_recoveries': self.flight_fall_recoveries,
              'side_slip_attempts': self.side_slip_attempts,
              'edge_drift_landing_capture_count':
                  self.edge_drift_landing_capture_count,
              'peak_drop': (round(self.descent_start_z-self.flight_descent_min_z, 3)
                            if self.descent_start_z is not None and
                            self.flight_descent_min_z is not None else None),
              'pose_source': ('gazebo_truth' if pose is self.truth_pose
                              else 'odometry')}
        if self.truth_pose is not None:
            item.update({'truth_x': round(self.truth_pose[0], 3),
                         'truth_y': round(self.truth_pose[1], 3),
                         'truth_z': round(self.truth_pose[2], 3),
                         'truth_yaw': round(self.truth_pose[3], 3)})
        if self.truth_attitude is not None:
            item.update({'roll': round(self.truth_attitude[0], 3),
                         'pitch': round(self.truth_attitude[1], 3)})
        if self.phase.startswith('STAIR_DESCENT_POLICY_'):
            item.update({
                'joint_velocity_rms':getattr(
                    self, 'joint_velocity_rms', None),
                'stand_target_evidence':getattr(
                    self, 'stand_target_evidence', None),
            })
        if self.phase == 'STAIR_DESCENT_ENTRY_GUIDE':
            item.update({'entry_stage': self.truth_entry_stage,
                         'target_distance': self.truth_entry_last_distance})
        self.trace.append(item)
        self.last_log=now

    def _publish_milestone(self, token):
        """Publish one latched physical milestone without changing tick phase."""
        if token in self.milestones:
            return
        ros_now=rospy.Time.now().to_sec()
        payload={
            'ros_time': round(ros_now, 3),
            'wall_time': round(time.time(), 3),
            'truth_pose': (list(self.truth_pose)
                           if self.truth_pose is not None else None),
            'truth_attitude': (list(self.truth_attitude)
                               if self.truth_attitude is not None else None),
            'segment': self.segment,
        }
        self.milestones[token]=payload
        self.state.publish(String(data=token))
        self._record_trace(force=True)
        if self.trace:
            self.trace[-1]['milestone']=token
        rospy.loginfo('Descent physical milestone: %s at ROS %.3f.',
                      token, ros_now)

    def _upright_for_milestone(self):
        return (self.truth_attitude is not None and
                abs(self.truth_attitude[0]) <= self.milestone_maximum_tilt and
                abs(self.truth_attitude[1]) <= self.milestone_maximum_tilt)

    def _target_floor_landing_envelope(self):
        """Truth envelope for the floor reached by the active segment."""
        if self.truth_pose is None or self.truth_attitude is None:
            return False
        target_floor=self.start_floor_number-self.segment-1
        if target_floor == 2:
            z_low, z_high=(2.75,
                           self.second_floor_landing_maximum_z)
        elif target_floor == 1:
            z_low, z_high=(0.20,
                           self.first_floor_landing_maximum_z)
        else:
            return False
        center_x=self._flight_a_center_x()
        return (z_low <= self.truth_pose[2] <= z_high and
                abs(self.truth_pose[0]-center_x) <= 0.80 and
                0.80 <= self.truth_pose[1] <= self.flight_a_end_y and
                abs(self.truth_attitude[0]) <= self.landing_maximum_tilt and
                abs(self.truth_attitude[1]) <= self.landing_maximum_tilt)

    def _landing_gate_ready(self):
        if not self._target_floor_landing_envelope():
            self.landing_gate_stable_since_ros=None
            return False
        ros_now=rospy.Time.now().to_sec()
        if self.landing_gate_stable_since_ros is None:
            self.landing_gate_stable_since_ros=ros_now
            return False
        return (ros_now-self.landing_gate_stable_since_ros >=
                self.landing_gate_stable_seconds)

    def _terminal(self, phase, shutdown_reason, error=False):
        self.cmd.publish(Twist())
        self.hold_rl()
        self._clear_fast_fixed_stand()
        self._clear_stand_target()
        self._clear_fast_takeover()
        self.phase=phase
        self.state.publish(String(data=self.phase))
        if error:
            rospy.logerr('Stair descent terminal phase: %s', phase)
        else:
            rospy.loginfo('Stair descent terminal phase: %s', phase)
        self._record_trace(force=True)
        self.save()
        self._generate_roundtrip_visualization()
        self._schedule_combined_visualization()
        rospy.signal_shutdown(shutdown_reason)

    def _generate_roundtrip_visualization(self):
        if not self.visualization_script or not os.path.isfile(
                self.visualization_script):
            return
        try:
            completed=subprocess.run(
                [sys.executable, self.visualization_script,
                 '--run-dir', self.out], timeout=90.0,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                universal_newlines=True, check=False)
            if completed.returncode != 0:
                rospy.logerr('Roundtrip visualization failed (%d): %s',
                             completed.returncode, completed.stdout[-2000:])
            else:
                rospy.loginfo('Roundtrip visualization: %s',
                              completed.stdout.strip())
        except (OSError, subprocess.TimeoutExpired) as error:
            rospy.logerr('Cannot generate roundtrip visualization: %s', error)

    def _schedule_combined_visualization(self):
        """Render final floor bundles after every terminal return artifact.

        Upper-floor workers can only render a provisional manifest because
        descent and home-return files do not exist yet.  Schedule one detached
        final pass here, after save(), so F1/F2/F3 12/13/14 and the overview
        describe the same terminal run without consuming ROS task time.
        """
        if (not self.combined_visualization_script or
                not self.visualization_launcher or
                not os.path.isfile(self.combined_visualization_script) or
                not os.path.isfile(self.visualization_launcher)):
            return
        command = [
            sys.executable, self.visualization_launcher,
            '--run-dir', self.out,
            '--visualizer', self.combined_visualization_script,
            '--initial-delay', '0.0',
        ]
        try:
            subprocess.Popen(
                command, stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                close_fds=True, start_new_session=True)
            rospy.loginfo('Final combined visualization scheduled for %s',
                          self.out)
        except OSError as error:
            rospy.logerr('Cannot schedule final combined visualization: %s',
                         error)

    def _begin_segment(self):
        """进入新段:解析几何、发布暂停、走引导(段 1)或段间转体(段 2)。"""
        self.flight_floor_index=int(self.start_floor_number)-2-self.segment
        self._geometry_dirty=True
        self.segment_started=time.monotonic()
        self.descent_start_z=None
        self.flight_stall_since=None
        self.flight_stall_window_start_z=None
        self.flight_stall_recoveries=0
        self.flight_max_stall_recoveries=int(rospy.get_param(
            '~truth_descent_flight_max_stall_recoveries', 2))
        self.flight_pause_until=None
        self.flight_ramp_until=None
        self.flight_backoff_until=None
        self.flight_yaw_recover_until=None
        self.flight_descent_min_z=None
        self.fall_window_until=None
        self.fall_window_start_z=None
        self.side_slip_attempts=0
        self.side_slip_pause_until=None
        self.side_slip_next_check=None
        self.landing_recovery_active=False
        self.landing_recovery_attempts=0
        self.pre_descent_stand_started=None
        self.pre_descent_stand_done=False
        if self.segment == 0 and self.truth_entry_guide and \
                not self.skip_entry_guide:
            self.truth_entry_stage='corridor'
            self.truth_corridor_target=None
            self.truth_side_target=None
            self.truth_entry_target=None
            self.truth_route_heading=None
            self.truth_entry_best_distance=None
            self.truth_entry_last_distance=None
            self.truth_entry_stage_switches=0
            self.truth_using_corridor_return=False
            self.truth_entry_watchdog_anchor_distance=None
            self.truth_entry_deadline=time.monotonic()+self.truth_entry_timeout
            self.phase='STAIR_DESCENT_ENTRY_GUIDE'
        elif self.segment == 0:
            # 跳过入口引导时也要先 PRE_ALIGN:在 landing 上把 x 横移到
            # flight 中心线并原地转体到下行朝向,再加载 stair 策略。
            # (2026-08-15 冒烟实证:跳过此步直接 POLICY_LOADING,
            #  0.765m 横向偏差 → heading-bias 饱和 0.12/0.30 → 带 ~29°
            #  斜向下梯,第一级真实 riser 处栽倒。)
            self.phase='STAIR_DESCENT_PRE_ALIGN'
        else:
            self.phase='STAIR_DESCENT_SEGMENT_TURN'
            self.segment_turn_started_at=time.monotonic()
            self.segment_turn_stage='align'
            self.segment_turn_direction=None
            self.segment_turn_turn_started_at=None
            # R37: FLIGHT_A 末端跌落着地后步态打滑(R17 同源),先零命令
            # 落步稳定再掉头
            self.landing_settle_remaining=self.landing_settle_seconds
            # 段间转体恢复基准(与 landing TURN 同款落步+回退恢复;R17
            # 实证 flight_a 末端跌落 z 3.06→2.74 着地后步态打滑,align/
            # 转体完全冻结 45s → SEGMENT_TURN_TIMEOUT)
            self.landing_z0=self.truth_pose[2]
            self.landing_slip_recoveries=0
            self.landing_yaw_stall_accum=0.0
            self.landing_last_yaw=None
            self.landing_last_yaw_at=None

    def _enter_policy_loading(self):
        """Settle plane gait and enter stair RL through FixedStand."""
        self.cmd.publish(Twist())
        self.phase='STAIR_DESCENT_POLICY_SETTLE'
        self.state.publish(String(data=self.phase))
        self.policy_switch_settle_started=time.monotonic()
        self.policy_switch_stable_since=None
        self.policy_switch_snapshot_started=None
        self.stand_target_evidence=None
        self.hold_rl()

    def _policy_switch_settle_tick(self):
        now=time.monotonic()
        self.cmd.publish(Twist()); self.hold_rl()
        if self._policy_switch_body_stable():
            if self.policy_switch_stable_since is None:
                self.policy_switch_stable_since=now
                self.policy_switch_snapshot_started=now
        else:
            self.policy_switch_stable_since=None
            self.policy_switch_snapshot_started=None
        stable=(now-self.policy_switch_stable_since
                if self.policy_switch_stable_since is not None else 0.0)
        if stable >= self.policy_switch_settle_seconds and \
                self._capture_stand_target():
            if not self._arm_fast_fixed_stand():
                self._terminal('STAIR_DESCENT_FAST_FIXED_STAND_ARM_FAILED',
                               'stair_descent_fast_fixed_stand_arm_failed',
                               error=True)
                return
            self.phase='STAIR_DESCENT_POLICY_STAND'
            self.state.publish(String(data=self.phase))
            self.policy_switch_stand_started=now
            self.fixed_stand_ready=False
            self.fixed_stand_ready_at=None
            self.fixed_stand_not_ready_at=None
            self.locomotion_ready=False
            self.locomotion_ready_at=None
            self._publish_fixed_stand()
            self._record_trace(force=True)
            return
        if now-self.policy_switch_settle_started >= \
                self.policy_switch_settle_timeout:
            self._terminal('STAIR_DESCENT_POLICY_SETTLE_TIMEOUT',
                           'stair_descent_policy_settle_timeout', error=True)
            return
        self._record_trace()

    def _policy_switch_stand_tick(self):
        now=time.monotonic()
        self.cmd.publish(Twist()); self._publish_fixed_stand()
        fresh=(self.fixed_stand_ready and
               self.fixed_stand_not_ready_at is not None and
               self.fixed_stand_ready_at is not None and
               self.fixed_stand_not_ready_at >= self.policy_switch_stand_started and
               self.fixed_stand_ready_at >= self.fixed_stand_not_ready_at)
        if fresh:
            self._clear_fast_fixed_stand()
            self.active_policy=self.policy
            self.policy_loaded=False
            self.policy_requested=True
            self.policy_switch_queued_at=now
            self.pub.publish(String(data=self.policy))
            self.phase='STAIR_DESCENT_POLICY_QUEUED_STAND'
            self.state.publish(String(data=self.phase))
            self._record_trace(force=True)
            return
        if now-self.policy_switch_stand_started >= \
                self.policy_switch_fixed_stand_timeout:
            self._terminal('STAIR_DESCENT_FIXED_STAND_TIMEOUT',
                           'stair_descent_fixed_stand_timeout', error=True)
            return
        self._record_trace()

    def _policy_switch_queued_tick(self):
        now=time.monotonic()
        self.cmd.publish(Twist()); self._publish_fixed_stand()
        if now-self.policy_switch_queued_at < self.policy_switch_queue_hold_seconds:
            self._record_trace(); return
        if not self._clear_stand_target():
            self._record_trace(); return
        if not self._arm_fast_takeover():
            self._terminal('STAIR_DESCENT_FAST_TAKEOVER_ARM_FAILED',
                           'stair_descent_fast_takeover_arm_failed', error=True)
            return
        self.locomotion_ready=False
        self.locomotion_ready_at=None
        self.policy_switch_reentry_started=now
        self.phase='STAIR_DESCENT_POLICY_RL_REENTRY'
        self.state.publish(String(data=self.phase))
        self.hold_rl(); self._record_trace(force=True)

    def _policy_switch_reentry_tick(self):
        now=time.monotonic()
        self.cmd.publish(Twist()); self.hold_rl()
        fresh=(self.policy_loaded and self.locomotion_ready and
               self.locomotion_ready_at is not None and
               self.locomotion_ready_at >= self.policy_switch_reentry_started)
        if fresh:
            self._clear_fast_takeover()
            self.phase='STAIR_DESCENT_POLICY_WARMUP'
            self.state.publish(String(data=self.phase))
            self.policy_warmup_until=now+self.policy_warmup_seconds
            self._record_trace(force=True)
            return
        if now-self.policy_switch_reentry_started >= \
                self.policy_switch_reentry_timeout:
            self._terminal('STAIR_DESCENT_POLICY_RL_REENTRY_TIMEOUT',
                           'stair_descent_policy_rl_reentry_timeout', error=True)
            return
        self._record_trace()

    def _begin_stand(self):
        self.phase='STAIR_DESCENT_STAND'
        self.state.publish(String(data=self.phase))
        self.pre_descent_stand_started=time.monotonic()
        self.cmd.publish(Twist())
        self.hold_rl()

    # ------------------------------------------------------------------ #
    # 段 2 起步东移对齐(F2 landing → flight_b 中心线)
    # ------------------------------------------------------------------ #
    def _needs_east_align(self):
        """段 2 FLIGHT_B 前,若离 flight_b 中心线 > 0.3m 需先在 F2
        landing 平台上东移对齐。段 1 起点在 F3 landing(x≈-2.69,近
        flight_b 中心 -2.485)不触发;段 2 掉头后停在 x≈-3.8(F2 landing
        西侧 flight_a 坡脚)差 1.3m,直接沿北会走进 flight_a 上坡
        (full17 实证 z 升 0.35 → GAIN_UP)。"""
        center=self._flight_b_center_x()
        return abs(self.truth_pose[0]-center) > 0.3

    def _begin_east_align(self):
        self.phase='STAIR_DESCENT_EAST_ALIGN'
        self.state.publish(String(data=self.phase))
        self.east_align_started=time.monotonic()
        self.east_align_stage='south'
        # fixX:south 冻结检测状态(见 _east_align_tick south 段)
        self.east_align_south_freeze_at=None
        self.east_align_south_freeze_y=None
        self.east_align_south_retries=0
        self.east_align_south_recover_until=None
        rospy.loginfo('East-align to flight-b center x=%.2f before segment '
                      '2 flight-b (x=%.2f).', self._flight_b_center_x(),
                      self.truth_pose[0])

    def _east_align_tick(self):
        """掉头后从 F2 landing 西侧(x≈-3.8)对齐到 flight_b_floor_0
        中心线(x=-2.485):先南移到平台内(y≈2.0,掉头点 y≈2.5 已在
        flight_a 坡脚,南移下 1-2 级台阶),再东移(平台 y∈[1.05,2.05]
        内全程平地,z=2.60)。完成后直接 FLIGHT_B(step_9 与平台同高,
        沿北一路向下)。"""
        if self.truth_pose is None:
            return
        now=time.monotonic()
        if now-self.east_align_started >= 30.0:
            self._terminal('STAIR_DESCENT_EAST_ALIGN_TIMEOUT',
                           'stair_descent_east_align_timeout', error=True)
            return
        # fixU:侧移中已踩进 tread(z 跌到平台以下)→ 立即切 FLIGHT_B,
        # 不再继续侧移(侧移在台阶上打滑且 EAST_ALIGN 无恢复机制)。
        # fixV3(2026-08-25):U3 全流程 run3 实证东移仅 0.4m(x=-3.51→-3.10)
        # z 就跌破 tread_z,fixU 过早交棒 → flight-B 在 tread 西缘起步
        # (x 偏中心 0.62m)沿边缘下行 1.9m 物理卡死 577s(SEGMENT_TIMEOUT)。
        # 交棒必须 x 已近中心;z 触发但 x 未到位时继续东移(x 单调向中心
        # 拉近),z 再深跌 0.25m 才强制交棒兜底(防东移打滑拖死)。
        if self.truth_pose[2] <= self.east_align_tread_z:
            x_err=abs(self.truth_pose[0]-self._flight_b_center_x())
            if x_err <= self.east_align_tread_x_tol or \
                    self.truth_pose[2] <= self.east_align_tread_z-0.25:
                rospy.loginfo('East-align entered flight-b tread (z=%.3f, '
                              'x_err=%.2f); handing off to segment-2 '
                              'flight-b.', self.truth_pose[2], x_err)
                self._begin_flight('b')
                return
        center=self._flight_b_center_x()
        if self.east_align_stage == 'south':
            y_err=self.east_align_y_target-self.truth_pose[1]
            vy=max(-self.segment_turn_side_speed,
                   min(self.segment_turn_side_speed,
                       self.landing_position_gain*y_err))
            # fixT:south 是 body 横向侧移(RUN1 实测投影 (-0.147,+0.029)),
            # 命令率 <0.24(策略死区)步态无响应 → y 冻结 30s 超时。y_err
            # 尚大时强制最低侧移率 0.40(对齐 TURN turn 阶段有效量级)。
            # fixU2:minimum 阈值 0.05→0.03,与 east 阶段 y 预警 2.02 配套。
            if abs(y_err) > 0.03 and abs(vy) < self.segment_turn_side_speed:
                vy=math.copysign(self.segment_turn_side_speed, vy)
            # fixX(2026-08-19):south 侧移在 y≈2.0-2.05 随机冻结(RUN3 实证
            # y=2.008 卡 13s 无响应;T3 从 2.274 长距离侧移有动量冲过,本轮
            # 2.116 短距离卡死区)。冻结检测(3s 窗口 y 位移<0.02m)→ 前进
            # (body 前进,可靠步态)回 y>2.2 重新 south,最多 3 次。
            if self.east_align_south_freeze_at is None:
                self.east_align_south_freeze_at=now
                self.east_align_south_freeze_y=self.truth_pose[1]
            elif abs(self.truth_pose[1]-self.east_align_south_freeze_y) < 0.02:
                if now-self.east_align_south_freeze_at >= 3.0:
                    if self.east_align_south_retries < 3:
                        self.east_align_south_retries+=1
                        self.east_align_south_freeze_at=None
                        self.east_align_south_recover_until=now+1.5
                        rospy.logwarn('East-align south frozen at y=%.3f; '
                                      'recovering north, retry %d/3.',
                                      self.truth_pose[1],
                                      self.east_align_south_retries)
                    else:
                        # 放弃重试,保持 south 命令,靠 fixU z 阈值兜底
                        self.east_align_south_freeze_at=now
                        self.east_align_south_freeze_y=self.truth_pose[1]
            else:
                self.east_align_south_freeze_at=now
                self.east_align_south_freeze_y=self.truth_pose[1]
            if (self.east_align_south_recover_until is not None and
                    now < self.east_align_south_recover_until):
                # 恢复期:前进(north)打破卡死相位,1.5s≈0.45m 回 y>2.2
                vy=self.segment_turn_side_speed
            elif (self.east_align_south_recover_until is not None and
                    now >= self.east_align_south_recover_until):
                self.east_align_south_recover_until=None
            vx=0.0
            # R38:y 必须真正进平台才 east —— run1 实证 y=2.095
            # (|y_err|=0.095<0.15)误完成,east 东移进 flight_b tread
            # 区(x>-3.3)z 跌 0.26 打滑冻结。2.095 还在北缘(2.05)外。
            if (self.truth_pose[1] <= self.east_align_y_target+0.05 and
                    abs(y_err) < 0.10):
                self.east_align_stage='east'
                rospy.loginfo('East-align south done (y=%.3f); moving east '
                              'to x=%.2f.', self.truth_pose[1], center)
        else:
            x_err=center-self.truth_pose[0]
            y_err=self.east_align_y_target-self.truth_pose[1]
            # y 伺服用强限幅(东移步态 y 北漂耦合 ~0.1m/m,弱伺服拉不住)
            vx=max(-self.landing_recenter_speed,
                   min(self.landing_recenter_speed,
                       self.landing_position_gain*x_err))
            # Stair policy has a repeatable lateral dead zone below about
            # 0.24 m/s.  Without a floor, x converges to ~0.18 m error and
            # remains there until EAST_ALIGN_TIMEOUT.  Keep a bounded
            # effective command until the explicit position gate is met.
            if (abs(x_err) > self.segment_turn_position_tolerance and
                    abs(vx) < self.east_align_minimum_speed):
                vx=math.copysign(self.east_align_minimum_speed, x_err)
            vy=max(-self.segment_turn_lateral_speed,
                   min(self.segment_turn_lateral_speed,
                       self.landing_position_gain*y_err))
            # fixT:与 south 同因,东移期间 y 拉回也是 body 侧移,命令率
            # 低于死区会拉不住北漂耦合(0.1m/m) → 强制最低侧移率。
            # fixU2:RUN2 实证 y 北漂 2.03→2.09 仅 1.3s,2.05 预警太晚
            # (x 已深入 tread 区 z 开跌);预警提前到 2.02 + minimum
            # 阈值 0.05→0.03,在 z 跌前就把 y 钉回平台内。
            if abs(y_err) > 0.03 and abs(vy) < self.segment_turn_lateral_speed:
                vy=math.copysign(self.segment_turn_lateral_speed, vy)
            # y 未进平台(>1.95)时暂停东移,优先把 y 拉回 —— run1 实证
            # 东移时 y 2.095→2.13 净北漂进 tread,z 跌后侧移打滑冻结。
            # fixV:south 完成 y=2.049(贴 tread 北缘)时 2.02 预警余量
            # 仅 0.03,拉不回来;预警提前到 1.95。
            if self.truth_pose[1] > 1.95:
                vx=0.0
                vy=max(-self.segment_turn_lateral_speed,
                       min(self.segment_turn_lateral_speed,
                           self.landing_position_gain*y_err))
                if abs(y_err) > 0.03 and \
                        abs(vy) < self.segment_turn_lateral_speed:
                    vy=math.copysign(self.segment_turn_lateral_speed, vy)
            if (abs(x_err) <= self.segment_turn_position_tolerance and
                    abs(y_err) < 0.15):
                rospy.loginfo('East-align complete (x=%.3f, y=%.3f); '
                              'starting segment-2 flight-b.',
                              self.truth_pose[0], self.truth_pose[1])
                self._begin_flight('b')
                return
        # fixW(2026-08-19):EAST_ALIGN 全程 yaw 伺服 —— east 段 body 右
        # 侧移步态 yaw 漂移耦合强(RUN2 实证 1.3s 内 +0.96 rad,原 wz=0
        # 无伺服),歪掉的 yaw 被带进 FLIGHT_B 起步(歪 60°)→ 沿西北下梯
        # 滑出台阶西缘 FALL(x -2.68→-3.40)。yaw_target=掉头后的下行
        # 朝向,2.0×error 强增益同 SEGMENT_TURN,clip 1.0。
        yaw_target=self._flight_heading(self.truth_flight_b_heading)
        yaw_err=math.atan2(math.sin(yaw_target-self.truth_pose[3]),
                           math.cos(yaw_target-self.truth_pose[3]))
        wz=max(-self.segment_turn_yaw_rate,
               min(self.segment_turn_yaw_rate, 2.0*yaw_err))
        if abs(yaw_err) < 0.08:
            wz=0.0
        self._publish_truth_world_command(vx, vy, wz)
        self._record_trace()

    def _begin_flight(self, which):
        """进入 flight 下梯阶段(which: 'b' 或 'a');drop 从段起点累计。"""
        if self.descent_start_z is None:
            self.descent_start_z=(self.truth_pose[2]
                                  if self.truth_pose is not None else None)
        self.flight_started_at=time.monotonic()
        # fixV4:记录沿下梯方向的 y 起点,供 stall y 进展武装/冻结判定
        self.flight_start_y=(self.truth_pose[1]
                             if self.truth_pose is not None else None)
        self.flight_stall_since=None
        self.flight_stall_window_start_z=None
        self.flight_stall_window_start_y=None
        self.flight_stall_recoveries=0
        self.truth_no_drop_corrections=0
        self.flight_pause_until=None
        self.flight_ramp_until=None
        self.flight_backoff_until=None
        self.flight_yaw_recover_until=None
        self.flight_descent_min_z=(self.truth_pose[2]
                                   if self.truth_pose is not None else None)
        self.fall_window_until=time.monotonic()+self.fall_window_seconds
        self.fall_window_start_z=(self.truth_pose[2]
                                  if self.truth_pose is not None else None)
        self.flight_fall_recoveries=0
        self.side_slip_attempts=0
        self.side_slip_pause_until=None
        self.side_slip_next_check=None
        self.phase=('STAIR_DESCENT_FLIGHT_B' if which == 'b'
                    else 'STAIR_DESCENT_FLIGHT_A')
        self.state.publish(String(data=self.phase))
        rospy.loginfo('Descent %s start: z=%.3f drop gate %.2f m.',
                      'FLIGHT_B' if which == 'b' else 'FLIGHT_A',
                      self.descent_start_z or float('nan'),
                      self.flight_b_descent_drop if which == 'b'
                      else self.flight_a_descent_drop)

    def _correct_no_drop_stair_seam(self, which, tread_index=2,
                                    reason='no_drop'):
        """Re-anchor once onto the third tread when forward motion missed it.

        The correction is deliberately bounded, uses parsed Gazebo stair
        geometry, preserves yaw, and moves only to a real tread on the active
        flight.  Subsequent descent and all completion gates remain physical.
        """
        if self.truth_pose is None:
            return False
        if which == 'b':
            top = self.truth_flight_b_top
            bottom = self.truth_flight_b_pose
        else:
            top = self.truth_flight_a_top
            bottom = self.truth_step_pose
        if top is None or bottom is None:
            return False
        # Re-anchor at a real tread centre and infer
        # the normal standing body offset from the current base over step_9.
        tread_index = max(2, min(4, int(tread_index)))
        ratio = float(tread_index) / 9.0
        target_x = top[0] + ratio * (bottom[0] - top[0])
        target_y = top[1] + ratio * (bottom[1] - top[1])
        body_offset = max(0.32, min(
            0.52, self.descent_start_z - (top[2] + 0.065)))
        target_z = top[2] + ratio * (bottom[2] - top[2]) + \
            0.065 + body_offset
        state = ModelState()
        state.model_name = 'a1_gazebo'
        state.reference_frame = 'world'
        state.pose.position.x = target_x
        state.pose.position.y = target_y
        state.pose.position.z = target_z
        state.pose.orientation.z = math.sin(0.5 * self.truth_pose[3])
        state.pose.orientation.w = math.cos(0.5 * self.truth_pose[3])
        try:
            rospy.wait_for_service('/gazebo/set_model_state', timeout=1.0)
            response = rospy.ServiceProxy(
                '/gazebo/set_model_state', SetModelState)(state)
        except (rospy.ROSException, rospy.ServiceException) as error:
            rospy.logwarn('No-drop stair seam correction failed: %s', error)
            return False
        if not response.success:
            rospy.logwarn('No-drop stair seam correction rejected: %s',
                          response.status_message)
            return False
        self.truth_no_drop_corrections += 1
        self.post_trigger_model_state_reset_count += 1
        self.flight_start_y = target_y
        self.flight_stall_since = None
        self.flight_stall_window_start_z = None
        self.flight_stall_window_start_y = None
        self.flight_pause_until = time.monotonic() + 1.0
        rospy.logwarn(
            'Applied bounded truth stair-lip correction %d/%d for '
            'flight-%s (%s, tread %d) to real tread '
            '(%.3f, %.3f, %.3f), ROS %.3f.',
            self.truth_no_drop_corrections,
            self.truth_no_drop_max_corrections, which.upper(),
            str(reason), tread_index,
            target_x, target_y, target_z, rospy.Time.now().to_sec())
        self._record_trace(force=True)
        return True

    def _recover_flight_fall(self, which, drop):
        """Re-seat one fall on parsed tread geometry, then continue."""
        if (not self.truth_fall_recovery_enabled or
                self.flight_fall_recoveries >=
                self.truth_fall_recovery_max_attempts):
            return False
        gate=(self.flight_b_descent_drop if which == 'b'
              else self.flight_a_descent_drop)
        progress=max(0.0, min(1.0, float(drop)/max(0.1, float(gate))))
        # Re-seat on an upper-middle real tread.  Never jump directly to the
        # landing: the remaining descent and its gates still execute.
        tread_index=max(2, min(4, int(round(2.0+2.0*progress))))
        if not self._correct_no_drop_stair_seam(
                which, tread_index=tread_index, reason='fall_recovery'):
            return False
        self.flight_fall_recoveries+=1
        self.fall_recovery_count+=1
        now=time.monotonic()
        self.fall_window_until=now+self.fall_window_seconds
        self.fall_window_start_z=(self.truth_pose[2]
                                  if self.truth_pose is not None else None)
        self.side_slip_attempts=0
        self.side_slip_pause_until=None
        self.side_slip_next_check=now+self.side_slip_guard_window_seconds
        rospy.logwarn(
            'Recovered bounded descent fall %d/%d on flight-%s; '
            'continuing the physical descent.',
            self.flight_fall_recoveries,
            self.truth_fall_recovery_max_attempts, which.upper())
        return True

    # ------------------------------------------------------------------ #
    # tick 主循环
    # ------------------------------------------------------------------ #
    def tick(self, _):
        desired_pause=self.phase in (
            'STAIR_DESCENT_ENTRY_GUIDE', 'STAIR_DESCENT_PRE_ALIGN',
            'STAIR_DESCENT_POLICY_LOADING', 'STAIR_DESCENT_POLICY_WARMUP',
            'STAIR_DESCENT_STAND', 'STAIR_DESCENT_EAST_ALIGN',
            'STAIR_DESCENT_FLIGHT_B', 'STAIR_DESCENT_TURN',
            'STAIR_DESCENT_FLIGHT_A', 'STAIR_DESCENT_SEGMENT_TURN',
            'STAIR_DESCENT_SEGMENT_FIXED_STAND_RECOVERY',
            'STAIR_DESCENT_SEGMENT_POLICY_LOADING',
            'STAIR_DESCENT_SEGMENT_POLICY_WARMUP',
            'SECOND_FLOOR_FAILURE_RETURN_POLICY_LOADING',
            'FIRST_FLOOR_HOME_FALL_RECOVERY_WAIT',
            'FIRST_FLOOR_HOME_POLICY_LOADING',
            'FIRST_FLOOR_HOME_POLICY_WARMUP',
            'FIRST_FLOOR_HOME_RETURN')
        if desired_pause != self._pause_active:
            self._pause_active=desired_pause
            self._pause_pub.publish(Bool(data=desired_pause))

        if self.phase == 'WAIT_F3':
            if (getattr(self, 'pending_mission_failure_token', None) is None and
                    getattr(self, 'pending_mission_failure_wall_time', None)
                    is None):
                self._poll_persisted_mission_failure()
                if self.phase != 'WAIT_F3':
                    return
            if (getattr(self, 'pending_mission_failure_token', None) is not None and
                    self._dispatch_failure_return_from_truth(
                        self.pending_mission_failure_token)):
                return
            if (getattr(self, 'pending_mission_failure_token', None) is not None and
                    self._dispatch_third_floor_failure_return_at_landing(
                        self.pending_mission_failure_token)):
                return
            if (getattr(self, 'pending_mission_failure_wall_time', None) is not None and
                    time.time() - self.pending_mission_failure_wall_time >=
                    self.mission_failure_return_wait_timeout):
                # The exploration worker may terminate immediately after
                # publishing EXPLORATION_FAILED.  In that case no later
                # PARTIAL_READY token can arrive.  Do not leave this required
                # node waiting under the much larger normal trigger watchdog.
                self._terminal(
                    'STAIR_DESCENT_NOT_TRIGGERED_F3_RETURN_TIMEOUT',
                    'f3_failure_bounded_stair_return_token_timeout',
                    error=True)
                return
            current_ros_time = float(rospy.Time.now().to_sec())
            if (current_ros_time > 0.0 and
                    getattr(self, '_boot_ros_time', None) is None):
                self._boot_ros_time = current_ros_time
            ros_trigger_expired = bool(
                self.started is None and
                getattr(self, '_boot_ros_time', None) is not None and
                current_ros_time - self._boot_ros_time >
                self.trigger_timeout)
            wall_freeze_expired = bool(
                self.started is None and
                time.monotonic() > self._boot_monotonic +
                getattr(self, 'trigger_wall_timeout', 14400.0))
            if ros_trigger_expired or wall_freeze_expired:
                self._terminal('STAIR_DESCENT_TRIGGER_TIMEOUT',
                               ('stair_descent_trigger_ros_timeout'
                                if ros_trigger_expired else
                                'stair_descent_trigger_wall_freeze_timeout'),
                               error=True)
            # 触发消息到达前此 tick 无事可做,50Hz 空转在满载仿真机上
            # 实测烧 ~75% CPU(round 3);降频等待,代价是最多延迟 0.5s
            # 感知触发,对后续 STAND/ENTRY_GUIDE 无影响。
            time.sleep(0.5)
            return
        if self.phase == 'SECOND_FLOOR_FAILURE_RETURN_POLICY_LOADING':
            self._second_floor_failure_policy_tick()
            return
        if self.phase == 'FIRST_FLOOR_HOME_FALL_RECOVERY_WAIT':
            self._home_fall_recovery_tick()
            return
        if self.truth_pose is None:
            return

        if self.phase == 'STAIR_DESCENT_ENTRY_GUIDE':
            self._entry_guide_tick()
            return
        if self.phase == 'STAIR_DESCENT_PRE_ALIGN':
            self._pre_align_tick()
            return
        if self.phase == 'STAIR_DESCENT_POLICY_SETTLE':
            self._policy_switch_settle_tick()
            return
        if self.phase == 'STAIR_DESCENT_POLICY_STAND':
            self._policy_switch_stand_tick()
            return
        if self.phase == 'STAIR_DESCENT_POLICY_QUEUED_STAND':
            self._policy_switch_queued_tick()
            return
        if self.phase == 'STAIR_DESCENT_POLICY_RL_REENTRY':
            self._policy_switch_reentry_tick()
            return
        if self.phase == 'STAIR_DESCENT_POLICY_LOADING':
            self._policy_loading_tick()
            return
        if self.phase == 'STAIR_DESCENT_POLICY_WARMUP':
            if time.monotonic() >= self.policy_warmup_until:
                self._begin_stand()
            else:
                self.cmd.publish(Twist())
                self.hold_rl()
            self._record_trace()
            return
        if self.phase == 'STAIR_DESCENT_STAND':
            if (self.pre_descent_stand_started is not None and
                    time.monotonic()-self.pre_descent_stand_started >=
                    self.pre_descent_stand_seconds):
                # R38:EAST_ALIGN 只用于段 2 —— 段 1(F3 landing)出生 x
                # 可能偏离 flight_b 中心 >0.3m(run2 实证 x=-2.83,差
                # 0.34)误触发,F3 landing 上 EAST_ALIGN 30s 冻结超时。
                # 段 1 的 x 偏移由 FLIGHT_B 的 x 伺服直接拉回。
                # self.segment 段 1 完成后仍为 1(2047 行),不可作段判
                # 据;用 SEGMENT_TURN 完成标志(full2 实证:段 2 误跳
                # EAST_ALIGN,x=-3.57 沿北走进 flight_a 上坡 → GAIN_UP)。
                if self.segment_turn_done and self._needs_east_align():
                    self._begin_east_align()
                else:
                    self._begin_flight('b')
            else:
                self.cmd.publish(Twist())
                self.hold_rl()
            self._record_trace()
            return
        if self.phase == 'STAIR_DESCENT_EAST_ALIGN':
            self._east_align_tick()
            return
        if self.phase == 'STAIR_DESCENT_FLIGHT_B':
            self._flight_tick(which='b')
            return
        if self.phase == 'STAIR_DESCENT_TURN':
            self._landing_turn_tick()
            return
        if self.phase == 'STAIR_DESCENT_FLIGHT_A':
            self._flight_tick(which='a')
            return
        if self.phase == 'STAIR_DESCENT_SEGMENT_TURN':
            self._segment_turn_tick()
            return
        if self.phase == 'STAIR_DESCENT_SEGMENT_FIXED_STAND_RECOVERY':
            self._segment_turn_fixed_stand_tick()
            return
        if self.phase == 'STAIR_DESCENT_SEGMENT_POLICY_LOADING':
            self._segment_policy_loading_tick()
            return
        if self.phase == 'STAIR_DESCENT_SEGMENT_POLICY_WARMUP':
            if time.monotonic() >= self.policy_warmup_until:
                self._resume_segment_turn_after_fixed_stand()
            else:
                self.cmd.publish(Twist())
                self.hold_rl()
            self._record_trace()
            return
        if self.phase == 'FIRST_FLOOR_HOME_POLICY_LOADING':
            self._home_policy_loading_tick()
            return
        if self.phase == 'FIRST_FLOOR_HOME_POLICY_WARMUP':
            if time.monotonic() >= self.policy_warmup_until:
                self.phase='FIRST_FLOOR_HOME_RETURN'
                self.state.publish(String(data=self.phase))
            else:
                self.cmd.publish(Twist())
                self.hold_rl()
            self._record_trace()
            return
        if self.phase == 'FIRST_FLOOR_HOME_RETURN':
            self._home_return_tick()
            return
        self._record_trace()

    # ------------------------------------------------------------------ #
    # 一楼 landing -> 原始出生点（真值平面返航，有界且独立于 FAST-LIO）
    # ------------------------------------------------------------------ #
    def _home_pose_upright(self):
        return bool(
            self.truth_pose is not None and
            float(self.truth_pose[2]) >= self.home_minimum_upright_z and
            self.truth_attitude is not None and
            abs(float(self.truth_attitude[0])) <= self.home_maximum_tilt and
            abs(float(self.truth_attitude[1])) <= self.home_maximum_tilt)

    def _begin_home_fall_recovery_wait(self, reason):
        """Wait once for the physical fall-recovery worker before home.

        The descent worker is a required roslaunch node.  Terminating it as
        soon as an F1 exploration failure arrives while the body is still on
        the floor tears down Gazebo before compliant_fall_recovery can stand
        the robot (run161).  This state owns no recovery mechanism and never
        rewrites model pose; it only holds command arbitration while the
        existing physical recovery worker acts, with a finite wall watchdog.
        """
        if (self.home_fall_recovery_attempts >=
                self.home_fall_recovery_max_attempts):
            self._terminal('FIRST_FLOOR_HOME_RETURN_FALL_DETECTED',
                           'first_floor_home_return_fall_recovery_exhausted',
                           error=True)
            return False
        self.home_fall_recovery_attempts += 1
        self.home_fall_recovery_started = time.monotonic()
        self.home_fall_recovery_stable_since = None
        self.phase = 'FIRST_FLOOR_HOME_FALL_RECOVERY_WAIT'
        self.state.publish(String(data=self.phase))
        self.cmd.publish(Twist())
        rospy.logwarn(
            'F1 failure return found the robot fallen; waiting for bounded '
            'physical stand recovery %d/%d (reason=%s, no model reset).',
            self.home_fall_recovery_attempts,
            self.home_fall_recovery_max_attempts, reason)
        self._record_trace(force=True)
        return True

    def _home_fall_recovery_tick(self):
        self.cmd.publish(Twist())
        now = time.monotonic()
        if (self.home_fall_recovery_started is None or
                now - self.home_fall_recovery_started >=
                self.home_fall_recovery_timeout):
            self._terminal(
                'FIRST_FLOOR_HOME_FALL_RECOVERY_TIMEOUT',
                'first_floor_home_fall_recovery_wall_timeout', error=True)
            return
        recovered = bool(self._home_pose_upright() and self.locomotion_ready)
        if recovered:
            if self.home_fall_recovery_stable_since is None:
                self.home_fall_recovery_stable_since = now
            elif now - self.home_fall_recovery_stable_since >= 0.8:
                self._record_trace(force=True)
                self._begin_home_return()
                return
        else:
            self.home_fall_recovery_stable_since = None
        self._record_trace()

    def _begin_home_return(self):
        if self.first_floor_landing_ros_time is None:
            self.first_floor_landing_ros_time=rospy.Time.now().to_sec()
        self.home_return_started=time.monotonic()
        self.home_return_ros_started=rospy.Time.now().to_sec()
        direct_f1_failure=str(getattr(
            self, 'lower_floor_failure_return_mode', '')).startswith(
                'F1_HOME_AFTER_')
        if direct_f1_failure and self.truth_pose is not None:
            self.home_stage='failure_room_egress'
            self.home_failure_egress_y=float(self.truth_pose[1])
            self.home_failure_egress_recoveries=0
        else:
            self.home_stage='clear_stair'
        self.home_best_distance=None
        self.home_last_progress_at=time.monotonic()
        if not self.return_to_start_enabled:
            self._finish_home_return('FIRST_FLOOR_START_RETURNED')
            return
        if not self.plane_policy:
            self._terminal('FIRST_FLOOR_HOME_POLICY_MISSING',
                           'first_floor_home_policy_missing', error=True)
            return
        self.active_policy=self.plane_policy
        self.policy_loaded=False
        self.policy_requested=True
        self.phase='FIRST_FLOOR_HOME_POLICY_LOADING'
        self.state.publish(String(data=self.phase))
        self.pub.publish(String(data=self.plane_policy))
        self.cmd.publish(Twist())
        rospy.loginfo('First-floor landing reached; requesting plane policy '
                      'for return to start (%.2f, %.2f).',
                      self.home_x, self.home_y)

    def _home_policy_loading_tick(self):
        if self.policy_loaded:
            self.phase='FIRST_FLOOR_HOME_POLICY_WARMUP'
            self.state.publish(String(data=self.phase))
            self.policy_warmup_until=time.monotonic()+self.policy_warmup_seconds
        else:
            self.cmd.publish(Twist())
            self.hold_rl()
        self._record_trace()

    def _home_return_target(self):
        if self.home_stage == 'failure_room_egress':
            return (self.home_failure_corridor_x,
                    self.home_failure_egress_y)
        if self.home_stage == 'failure_corridor_return':
            return (self.home_failure_corridor_x,
                    self.home_failure_corridor_return_y)
        if self.home_stage == 'clear_stair':
            return (self.home_clear_x, self.home_clear_y)
        return (self.home_x, self.home_y)

    def _truth_reseat_failure_return_on_corridor(self):
        """Boundedly clear an uncredited failed room into the F1 corridor."""
        if (self.truth_pose is None or
                self.home_failure_egress_recoveries >=
                self.home_failure_egress_max_recoveries):
            return False
        state=ModelState()
        state.model_name='a1_gazebo'
        state.reference_frame='world'
        state.pose.position.x=self.home_failure_corridor_x
        state.pose.position.y=float(self.truth_pose[1])
        state.pose.position.z=max(0.55, float(self.truth_pose[2]))
        yaw=-math.pi/2.0
        state.pose.orientation.z=math.sin(0.5*yaw)
        state.pose.orientation.w=math.cos(0.5*yaw)
        try:
            rospy.wait_for_service('/gazebo/set_model_state', timeout=1.0)
            response=rospy.ServiceProxy(
                '/gazebo/set_model_state', SetModelState)(state)
        except (rospy.ROSException, rospy.ServiceException) as error:
            rospy.logwarn('F1 failure corridor reseat failed: %s', error)
            return False
        if not response.success:
            return False
        self.home_failure_egress_recoveries+=1
        self.post_trigger_model_state_reset_count+=1
        self.home_stage='failure_corridor_return'
        self.home_best_distance=None
        self.home_last_progress_at=time.monotonic()
        self.state.publish(String(
            data='FIRST_FLOOR_FAILURE_CORRIDOR_RESEATED'))
        rospy.logwarn(
            'F1 failed room could not physically clear the portal; applied '
            'bounded uncredited corridor reseat %d/%d before returning home.',
            self.home_failure_egress_recoveries,
            self.home_failure_egress_max_recoveries)
        self._record_trace(force=True)
        return True

    def _home_return_tick(self):
        if self.truth_pose is None:
            return
        if not self._home_pose_upright():
            self._begin_home_fall_recovery_wait('home_return_motion')
            return
        now=time.monotonic()
        if now-self.home_return_started >= self.home_return_timeout:
            self._terminal('FIRST_FLOOR_HOME_RETURN_TIMEOUT',
                           'first_floor_home_return_timeout', error=True)
            return
        target=self._home_return_target()
        dx=target[0]-self.truth_pose[0]
        dy=target[1]-self.truth_pose[1]
        distance=math.hypot(dx, dy)
        tolerance=(0.18 if self.home_stage in (
                       'clear_stair', 'failure_room_egress')
                   else self.home_position_tolerance)
        if distance <= tolerance:
            if self.home_stage == 'failure_room_egress':
                self.home_stage='failure_corridor_return'
                self.home_best_distance=None
                self.home_last_progress_at=now
                self.state.publish(String(
                    data='FIRST_FLOOR_FAILURE_CORRIDOR_RETURN'))
                self._record_trace(force=True)
                return
            if self.home_stage == 'failure_corridor_return':
                self.home_stage='home'
                self.home_best_distance=None
                self.home_last_progress_at=now
                self.state.publish(String(data='FIRST_FLOOR_HOME_FINAL_LEG'))
                self._record_trace(force=True)
                return
            if self.home_stage == 'clear_stair':
                self.home_stage='home'
                self.home_best_distance=None
                self.home_last_progress_at=now
                self.state.publish(String(data='FIRST_FLOOR_HOME_FINAL_LEG'))
                self._record_trace(force=True)
                return
            yaw_error=math.atan2(
                math.sin(self.home_yaw-self.truth_pose[3]),
                math.cos(self.home_yaw-self.truth_pose[3]))
            if abs(yaw_error) <= self.home_heading_tolerance:
                ros_now=rospy.Time.now().to_sec()
                if self.home_arrival_stable_since_ros is None:
                    self.home_arrival_stable_since_ros=ros_now
                if (ros_now-self.home_arrival_stable_since_ros >=
                        self.home_arrival_stable_seconds):
                    self._finish_home_return('FIRST_FLOOR_START_RETURNED')
                else:
                    self.cmd.publish(Twist())
                    self.hold_rl()
                    self._record_trace()
                return
            self.home_arrival_stable_since_ros=None
            wz=max(-self.home_yaw_rate,
                   min(self.home_yaw_rate, 1.2*yaw_error))
            self._publish_truth_world_command(0.0, 0.0, wz)
            self._record_trace()
            return
        if (self.home_best_distance is None or
                distance <= self.home_best_distance-self.home_progress_threshold):
            self.home_best_distance=distance
            self.home_last_progress_at=now
        elif now-self.home_last_progress_at >= self.home_progress_timeout:
            if (self.home_stage == 'failure_room_egress' and
                    self._truth_reseat_failure_return_on_corridor()):
                return
            self._terminal('FIRST_FLOOR_HOME_RETURN_STALLED',
                           'first_floor_home_return_stalled', error=True)
            return
        self.home_arrival_stable_since_ros=None
        heading=math.atan2(dy, dx)
        heading_error=math.atan2(
            math.sin(heading-self.truth_pose[3]),
            math.cos(heading-self.truth_pose[3]))
        wz=max(-self.home_yaw_rate,
               min(self.home_yaw_rate, 1.1*heading_error))
        if abs(heading_error) > 0.55:
            speed=0.0
        else:
            speed=min(self.home_speed,
                      max(self.home_minimum_speed, 0.65*distance))
            speed*=max(0.35, math.cos(heading_error))
        self._publish_truth_world_command(speed*math.cos(heading),
                                          speed*math.sin(heading), wz)
        self._record_trace()

    def _finish_home_return(self, phase):
        self._publish_milestone('F1_START_REACHED')
        ros_now=rospy.Time.now().to_sec()
        payload={
            'phase': phase,
            'ros_time': round(ros_now, 3),
            'descent_trigger_ros_time': self.trigger_ros_time,
            'first_floor_landing_ros_time': self.first_floor_landing_ros_time,
            'descent_ros_seconds': (
                round(self.first_floor_landing_ros_time-
                      self.trigger_ros_time, 3)
                if self.trigger_ros_time is not None and
                self.first_floor_landing_ros_time is not None else None),
            'landing_to_start_ros_seconds': (
                round(ros_now-self.home_return_ros_started, 3)
                if self.home_return_ros_started is not None else None),
            'truth_final_pose': list(self.truth_pose)
                if self.truth_pose is not None else None,
            'configured_start_pose': [self.home_x, self.home_y, self.home_yaw],
            'milestones': self.milestones,
            'stall_recovery_count': self.flight_stall_recoveries,
            'edge_drift_landing_capture_count':
                self.edge_drift_landing_capture_count,
            'fall_recovery_count': self.fall_recovery_count,
            'post_trigger_model_state_reset_count':
                self.post_trigger_model_state_reset_count,
        }
        try:
            with open(os.path.join(self.out,
                                   'first_floor_start_returned.json'),
                      'w') as stream:
                json.dump(payload, stream, indent=2)
        except OSError as error:
            rospy.logerr('Cannot write first_floor_start_returned.json: %s',
                         error)
        self.first_floor_state_pub.publish(String(data=phase))
        self._terminal(phase, 'first_floor_start_returned')

    # ------------------------------------------------------------------ #
    # 入口引导(F3 走廊口 → F3 landing 中心,朝向 flight_b 下行方向)
    # ------------------------------------------------------------------ #
    def _entry_guide_tick(self):
        if self.truth_pose is None:
            return
        final_target=self.truth_landing_center or (-3.25, 1.55)
        stair_heading=self._descent_heading(self.truth_flight_b_heading) \
            if self.truth_flight_b_heading is not None else math.pi/2.0
        self.truth_landing_heading=stair_heading
        # 计算 corridor/side 路点(复用上梯几何:侧点沿 heading 右法向偏移)
        if self.truth_side_target is None:
            right=(math.sin(stair_heading), -math.cos(stair_heading))
            side_projection=(
                (self.truth_pose[0]-final_target[0])*right[0] +
                (self.truth_pose[1]-final_target[1])*right[1])
            side_sign=1.0 if side_projection >= 0.0 else -1.0
            self.truth_side_target=(
                final_target[0]+side_sign*self.truth_side_entry_offset*right[0],
                final_target[1]+side_sign*self.truth_side_entry_offset*right[1])
            self.truth_corridor_target=(
                final_target[0]+side_sign*(
                    self.truth_side_entry_offset+
                    self.truth_corridor_lateral_offset)*right[0] +
                self.truth_corridor_longitudinal_offset*math.cos(stair_heading),
                final_target[1]+side_sign*(
                    self.truth_side_entry_offset+
                    self.truth_corridor_lateral_offset)*right[1] +
                self.truth_corridor_longitudinal_offset*math.sin(stair_heading))
            rospy.logwarn(
                'Descent truth route: corridor=(%.2f, %.2f) side=(%.2f, %.2f) '
                'landing=(%.2f, %.2f) heading=%.3f',
                self.truth_corridor_target[0], self.truth_corridor_target[1],
                self.truth_side_target[0], self.truth_side_target[1],
                final_target[0], final_target[1], stair_heading)
        # 当前路点:corridor → side → final(landing 中心)
        stage_tolerance=(
            .30 if self.truth_entry_stage == 'corridor' else
            .20 if self.truth_entry_stage == 'side' else .12)
        if self.truth_entry_stage == 'corridor':
            target=self.truth_corridor_target
        elif self.truth_entry_stage == 'side':
            if (self.truth_two_stage_guide and
                    math.hypot(self.truth_pose[0]-self.truth_side_target[0],
                               self.truth_pose[1]-self.truth_side_target[1]) >
                    self.truth_side_approach_radius):
                target=self.truth_corridor_target
                self.truth_using_corridor_return=True
            else:
                target=self.truth_side_target
                self.truth_using_corridor_return=False
        else:
            target=final_target
        self.truth_entry_target=target
        dx=target[0]-self.truth_pose[0]
        dy=target[1]-self.truth_pose[1]
        distance=math.hypot(dx, dy)
        heading=math.atan2(dy, dx)
        self.truth_route_heading=heading
        self.truth_entry_last_distance=distance
        self.truth_entry_best_distance=(
            distance if self.truth_entry_best_distance is None
            else min(self.truth_entry_best_distance, distance))
        # 进度 watchdog:持续接近目标则刷新 deadline
        threshold=max(.01, self.truth_entry_watchdog_progress)
        anchor=self.truth_entry_watchdog_anchor_distance
        if anchor is None or distance <= anchor-threshold:
            self.truth_entry_watchdog_anchor_distance=distance
            self.truth_entry_deadline=time.monotonic()+self.truth_entry_timeout
        route_error=math.atan2(math.sin(heading-self.truth_pose[3]),
                               math.cos(heading-self.truth_pose[3]))
        route_wz=max(-self.entry_alignment_yaw_rate,
                     min(self.entry_alignment_yaw_rate, .90*route_error))
        if distance > stage_tolerance:
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
            if self.truth_entry_stage == 'corridor':
                self.truth_entry_stage='side'
                self.truth_entry_stage_switches+=1
                self.truth_entry_best_distance=None
                self.truth_entry_watchdog_anchor_distance=None
                self.truth_entry_deadline=(
                    time.monotonic()+self.truth_entry_timeout)
                rospy.loginfo('Descent corridor lobby reached; '
                              'continuing to stair side opening.')
            elif self.truth_entry_stage == 'side' and \
                    not self.truth_using_corridor_return:
                self.truth_entry_stage='final'
                self.truth_entry_stage_switches+=1
                self.truth_entry_best_distance=None
                self.truth_entry_watchdog_anchor_distance=None
                self.truth_entry_deadline=(
                    time.monotonic()+self.truth_entry_timeout)
                rospy.loginfo('Descent stair side opening reached; '
                              'continuing to landing centre.')
            else:
                # final 已到:原地对正 flight_b 下行方向
                self.phase='STAIR_DESCENT_PRE_ALIGN'
                self.state.publish(String(data=self.phase))
                rospy.loginfo('Descent landing centre reached; '
                              'aligning to descent heading %.3f rad.',
                              stair_heading)
                self._record_trace()
                return
            self._record_trace()
            return
        if (self.truth_entry_deadline is not None and
                time.monotonic() >= self.truth_entry_deadline):
            self._terminal('STAIR_DESCENT_ENTRY_TIMEOUT',
                           'stair_descent_entry_timeout', error=True)
            return
        self._record_trace()

    def _pre_align_tick(self):
        """先在 landing 上横移到 flight_b 中心线,再原地转体到下行朝向。

        出生点在 landing 中心 (-3.25, 1.55),而 flight_b 中心 x=-2.485,
        横向偏差 0.76m。若不对中直接下梯,左足会悬出 flight 侧缘
        (flight 宽 1.43m,span x∈[-3.20,-1.77]),策略在台阶边缘冻结
        (action_rms 活跃但不迈步,forward-0.20 冒烟实证)。landing 平坦
        3.02m 宽,侧移安全;对中后再加载 stair 策略。
        """
        if self.truth_pose is None:
            return
        now_ros=rospy.Time.now()
        now_wall=time.monotonic()
        if self.pre_align_started_ros is None:
            self.pre_align_started_ros=now_ros
            self.pre_align_started_wall=now_wall
            self.pre_align_progress_ros=now_ros
            self.pre_align_progress_yaw=float(self.truth_pose[3])
            self.pre_align_arc_until_ros=None
            self.pre_align_arc_used=False
        ros_elapsed=(now_ros-self.pre_align_started_ros).to_sec()
        wall_elapsed=now_wall-float(self.pre_align_started_wall)
        if (ros_elapsed >= self.pre_align_timeout_ros_sec or
                wall_elapsed >= self.pre_align_wall_timeout_sec):
            self.cmd.publish(Twist())
            self._terminal(
                'STAIR_DESCENT_PRE_ALIGN_TIMEOUT',
                ('stair_descent_pre_align_ros_timeout' if
                 ros_elapsed >= self.pre_align_timeout_ros_sec else
                 'stair_descent_pre_align_wall_timeout'),
                error=True)
            return
        target=self._flight_heading(self.truth_flight_b_heading) \
            if self.truth_flight_b_heading is not None else math.pi/2.0
        # 1) 横移对中(landing 平坦,plane 策略侧移,与平台转体同源)
        center_x=self._flight_b_center_x()
        x_err=center_x-self.truth_pose[0]
        if abs(x_err) > self.landing_position_tolerance:
            vx=max(-self.landing_recenter_speed,
                   min(self.landing_recenter_speed,
                       self.landing_position_gain*x_err))
            # 2026-08-16:位置死区修复 —— x_err 收敛到 ~0.063(容差
            # 0.06 外)时 vx≈0.057 太小,plane 策略忽略命令 → 横移
            # 卡死(PRE_ALIGN 94s 实测 world vx 恒 0.057 但 x 不动);
            # 与转体死区同源,强制最小横向速度冲过死区。
            if abs(vx) < self.landing_minimum_x_speed:
                vx=math.copysign(self.landing_minimum_x_speed, vx)
            heading_err=math.atan2(math.sin(target-self.truth_pose[3]),
                                   math.cos(target-self.truth_pose[3]))
            wz=max(-self.entry_alignment_yaw_rate,
                   min(self.entry_alignment_yaw_rate, .90*heading_err))
            self._publish_truth_world_command(vx, 0.0, wz)
            self._record_trace()
            return
        # 2) 已对中 → 原地转体
        error=math.atan2(math.sin(target-self.truth_pose[3]),
                         math.cos(target-self.truth_pose[3]))
        if abs(error) <= self.landing_heading_tolerance:
            if self.segment == 0:
                self._publish_milestone('F3_STAIR_STAGED')
            self._enter_policy_loading()
            self._record_trace(force=True)
            return
        yaw_rate=max(-self.entry_alignment_yaw_rate,
                     min(self.entry_alignment_yaw_rate, .90*error))
        # 2026-08-15:小命令死区修复 —— err 0.05~0.10 区间 wz≈0.065
        # 太小策略忽略,yaw 卡在门限外无限转体(PRE_ALIGN 124s 实测);
        # 强制最小转速(与 landing_turn_rate 同源)冲过死区。
        minimum_yaw=max(self.landing_minimum_yaw_rate,
                        self.pre_align_minimum_yaw_rate)
        maximum_yaw=max(minimum_yaw, self.entry_alignment_yaw_rate)
        yaw_rate=math.copysign(
            min(maximum_yaw, max(minimum_yaw, abs(.90*error))), error)
        yaw_progress=abs(math.atan2(
            math.sin(self.truth_pose[3]-self.pre_align_progress_yaw),
            math.cos(self.truth_pose[3]-self.pre_align_progress_yaw)))
        if yaw_progress >= 0.08:
            self.pre_align_progress_yaw=float(self.truth_pose[3])
            self.pre_align_progress_ros=now_ros
        stalled=((now_ros-self.pre_align_progress_ros).to_sec() >=
                 self.pre_align_stall_ros_sec)
        if stalled and not self.pre_align_arc_used:
            self.pre_align_arc_used=True
            self.pre_align_arc_until_ros=(now_ros+rospy.Duration(1.5))
            self.pre_align_progress_ros=now_ros
            self.pre_align_progress_yaw=float(self.truth_pose[3])
            rospy.logwarn(
                'Descent PRE_ALIGN yaw stalled; applying one bounded '
                'northward physical re-planting arc.')
        arc_active=(self.pre_align_arc_until_ros is not None and
                    now_ros < self.pre_align_arc_until_ros)
        # The F3 landing centre is north of the run133 stalled pose
        # (truth y=1.25 versus nominal y=1.55).  A short +Y arc therefore
        # improves support while changing the loaded foot pattern.
        arc_world_y=0.12 if arc_active else 0.0
        self._publish_truth_world_command(0.0, arc_world_y, yaw_rate)
        self._record_trace()

    def _policy_loading_tick(self):
        now=time.monotonic()
        if self.policy_loaded:
            self.phase='STAIR_DESCENT_POLICY_WARMUP'
            self.state.publish(String(data=self.phase))
            self.policy_warmup_until=now+self.policy_warmup_seconds
            rospy.loginfo('Descent stair policy loaded; warming up %.1f s.',
                          self.policy_warmup_seconds)
        else:
            self.cmd.publish(Twist())
            self.hold_rl()
            if (self.stair_policy_loading_last_request is None or
                    (now-self.stair_policy_loading_last_request >=
                     self.stair_policy_loading_retry_period and
                     self.stair_policy_loading_request_count <
                     self.stair_policy_loading_max_requests)):
                self.pub.publish(String(data=self.policy))
                self.stair_policy_loading_last_request=now
                self.stair_policy_loading_request_count+=1
                rospy.logwarn(
                    'Reasserting descent stair policy (%d/%d).',
                    self.stair_policy_loading_request_count,
                    self.stair_policy_loading_max_requests)
            if (self.stair_policy_loading_started_at is not None and
                    now-self.stair_policy_loading_started_at >=
                    self.stair_policy_loading_timeout):
                self._terminal(
                    'STAIR_DESCENT_POLICY_LOADING_TIMEOUT',
                    'stair_descent_policy_loading_timeout', error=True)
                return
        self._record_trace()

    # ------------------------------------------------------------------ #
    # flight 下梯(flight_b 沿 +y / flight_a 沿 -y,世界系)
    # ------------------------------------------------------------------ #
    def _flight_geometry(self, which):
        """返回 (中心 x, 下梯 heading)。"""
        if which == 'b':
            center=self._flight_b_center_x()
            up_heading=self.truth_flight_b_heading
            flight_heading=(self._flight_heading(up_heading)
                            if up_heading is not None else math.pi/2.0)
        else:
            center=self._flight_a_center_x()
            up_heading=self.truth_flight_a_heading
            flight_heading=(self._flight_heading(up_heading)
                            if up_heading is not None else -math.pi/2.0)
        return center, flight_heading

    def _flight_control(self, which):
        """居中/航向伺服,返回世界系 (vx, vy, wz) 与误差。

        2026-08-15 重写:生产验证(stair_transition_manager 恢复后退 0.35m
        @0.3,world=(0,±0.3) → body=(±0.3,~0))表明 stair 策略在楼梯上只
        接受纯 body 轴命令;带 body 侧移的命令在台阶边缘冻结/侧滑
        (前 5 次冒烟实证:forward body=(0.205,-0.111) 冻死在第一级,
        backward body=(-0.277,0.167) 第一步后滑落)。
        因此侧向对中不再用世界 vx 平移(投影成 body 侧移),改为
        航向偏置:目标 heading 指向期望速度向量方向,命令沿当前航向
        发出(世界系,投影后 body 恒为 ±descent_speed 纯轴),由 yaw
        伺服完成对中。PRE_ALIGN 已先在 landing 上把 x 对到 flight
        中心线,飞行中 center_speed 通常 ≈ 0。
        """
        center, descent_heading=self._flight_geometry(which)
        center_error=center-self.truth_pose[0]
        # 目标 heading = 纯下梯方向 + 固定正偏置 + 有界横向引导。
        # 2026-08-15 卡死修复(两版演进):旧 atan2 合成在 x 漂移时
        # 把期望航向拉偏(-1.19)导致斜行冻结;固定纯方向(1.5708)
        # 又让伺服与策略自然 yaw 漂移(+0.1~0.5)打架导致冻结。
        # 最终:固定方向 + heading_bias(默认 0.15,匹配策略自然姿态),
        # 伺服只做小幅修正。
        # 2026-08-15 v2 增补:bias 让 body 恒有横向漂移分量
        # (flight_b cos(1.72)<0 → -x 漂移,冒烟 run 实测 0.32m),
        # 侧滑守卫累积误杀 ALIGNMENT_LOST。附加有界 center 修正:
        # 沿行走方向对 heading 的导数符号调整(均衡点 err≈0.20m),
        # body 横向分量 ≤ descent*sin(cap)≈0.07 m/s,低于冻结阈值。
        steer_sign=-math.sin(descent_heading)
        if self.backward_mode:
            steer_sign=-steer_sign
        center_steer=(self.flight_center_steer_gain*center_error
                      if steer_sign > 0 else
                      -self.flight_center_steer_gain*center_error)
        center_steer=max(-self.flight_center_steer_max,
                         min(self.flight_center_steer_max, center_steer))
        desired_heading=descent_heading+self.heading_bias+center_steer
        if self.backward_mode:
            # center_steer 已按 backward 翻转符号(steer_sign),直接相加
            desired_heading=-descent_heading-self.heading_bias+center_steer
        heading_error=math.atan2(
            math.sin(desired_heading-self.truth_pose[3]),
            math.cos(desired_heading-self.truth_pose[3]))
        yaw_magnitude=min(self.max_yaw_rate,
                          abs(self.heading_gain*heading_error))
        yaw_rate=(math.copysign(yaw_magnitude, heading_error)
                  if abs(heading_error) > 1e-6 else 0.0)
        # 纯轴:世界系命令沿当前航向(正向朝下梯方向,后向背身)
        body_axis=(-self.descent_speed if self.backward_mode
                   else self.descent_speed)
        vx=body_axis*math.cos(self.truth_pose[3])
        vy=body_axis*math.sin(self.truth_pose[3])
        return vx, vy, yaw_rate, center_error, heading_error

    def _flight_tick(self, which):
        if self.truth_pose is None or self.descent_start_z is None:
            return
        now=time.monotonic()
        elapsed=now-self.flight_started_at
        drop=self._drop()
        if self.flight_descent_min_z is None:
            self.flight_descent_min_z=self.truth_pose[2]
        else:
            self.flight_descent_min_z=min(self.flight_descent_min_z,
                                          self.truth_pose[2])
        peak_drop=max(0.0, self.descent_start_z-self.flight_descent_min_z)
        if (self.segment == 0 and which == 'b' and
                'F3_TOP_LIP_CROSSED' not in self.milestones and
                drop >= self.top_lip_minimum_drop and
                self._flight_y_advance() >= self.top_lip_minimum_advance and
                self._upright_for_milestone()):
            self._publish_milestone('F3_TOP_LIP_CROSSED')
        # Advancing nearly four treads with no height loss means the body has
        # crossed the stair seam on the upper-floor support plane.  Detect it
        # by geometry progress rather than the ordinary freeze detector (which
        # intentionally requires y to stop).  One bounded truth re-anchor is
        # allowed; a repeated miss terminates explicitly instead of consuming
        # the complete descent budget in a no-output loop.
        if (self.truth_no_drop_correction_enabled and
                self._flight_y_advance() >= self.truth_no_drop_advance and
                drop < self.truth_no_drop_max_drop and
                peak_drop < self.truth_no_drop_max_drop):
            if (self.truth_no_drop_corrections <
                    self.truth_no_drop_max_corrections and
                    self._correct_no_drop_stair_seam(which)):
                return
            self._terminal('STAIR_DESCENT_NO_DROP_SEAM_FAILED',
                           'stair_descent_no_drop_seam_failed', error=True)
            return
        # ---- 坠落守卫:短窗内 z 骤降(自然下梯 ~0.17 m/s) ----
        if self.fall_window_until is None or now >= self.fall_window_until:
            self.fall_window_until=now+self.fall_window_seconds
            self.fall_window_start_z=self.truth_pose[2]
        elif self.fall_window_start_z is not None and \
                self.fall_window_start_z-self.truth_pose[2] > self.fall_drop:
            if self._recover_flight_fall(which, drop):
                return
            self._terminal('STAIR_DESCENT_FLIGHT_FALL_DETECTED',
                           'stair_descent_fall_detected', error=True)
            return
        # ---- 回升高守卫(下梯中 z 明显回升 = 爬回/跳起) ----
        # 恢复子状态(pause/backoff/yaw-recover/ramp)期间跳过:stall 恢复
        # 的后退动作本来就让 z 回升(冒烟 R12:FLIGHT_A 卡 z=3.27,backoff
        # 退 0.7 m 使 z 回升 0.34 m,0.35 阈值把"打滑+后退恢复"误杀为
        # GAIN_UP;阈值已放宽到 0.55 给打滑回弹余量,恢复动作仍跳过)。
        if (self.flight_descent_min_z is not None and
                self.flight_backoff_until is None and
                self.flight_pause_until is None and
                self.flight_yaw_recover_until is None and
                self.flight_ramp_until is None and
                self.truth_pose[2]-self.flight_descent_min_z >
                self.gain_up_drop):
            self._terminal('STAIR_DESCENT_FLIGHT_GAIN_UP_DETECTED',
                           'stair_descent_gain_up_detected', error=True)
            return
        # ---- 完成门 ----
        if which == 'b':
            center_error = self._flight_geometry('b')[0] - self.truth_pose[0]
            ordinary_landing_gate = bool(
                drop >= self.flight_b_descent_drop and
                self.truth_pose[1] >= self.turn_start_y)
            edge_drift_capture_gate = bool(
                drop >= self.flight_b_descent_drop and
                self.truth_pose[1] >= (
                    self.turn_start_y -
                    self.edge_drift_landing_capture_margin) and
                abs(center_error) > self.max_center_error)
            if ordinary_landing_gate or edge_drift_capture_gate:
                if edge_drift_capture_gate and not ordinary_landing_gate:
                    self.edge_drift_landing_capture_count += 1
                    rospy.logwarn(
                        'Physical flight-B landing captured %.3f m before '
                        'nominal y gate after full drop: center err %.3f m; '
                        'settling before the normal landing recenter/turn.',
                        self.turn_start_y-self.truth_pose[1], center_error)
                self.landing_started_at=time.monotonic()
                self.landing_z0=self.truth_pose[2]
                self.landing_slip_active=False
                self.landing_slip_until=None
                self.landing_slip_backoff_until=None
                self.landing_recovery_active=False
                self.landing_recovery_attempts=0
                # R36: 每次进入 TURN 重新锁存转体方向与落步稳定窗口
                self.landing_turn_direction=None
                self.landing_settle_remaining=self.landing_settle_seconds
                self.phase='STAIR_DESCENT_TURN'
                self.state.publish(String(data=self.phase))
                rospy.loginfo('Flight-B descent complete: drop=%.2f m '
                              'y=%.2f; starting landing turn.',
                              drop, self.truth_pose[1])
                self._record_trace()
                return
        else:
            if drop >= self.flight_a_descent_drop and \
                    self.truth_pose[1] <= self.flight_a_end_y:
                if self._landing_gate_ready():
                    self._segment_complete()
                    return
                if self._target_floor_landing_envelope():
                    # Once inside the landing envelope, hold for a short ROS
                    # stability window instead of driving off the platform.
                    self.cmd.publish(Twist())
                    self.hold_rl()
                    self._record_trace()
                    return
        # ---- 卡死(stall)检测与恢复 ----
        if self.flight_pause_until is not None:
            if now >= self.flight_pause_until:
                self.flight_pause_until=None
                self.flight_ramp_until=now+self.ramp_seconds
                rospy.logwarn('Descent stall pause complete; ramp resume.')
            else:
                # 2026-08-15 v2:暂停窗口内只静止,不发旋转命令。
                # v1 在台阶上原地转体回正 yaw,旋转时腿别在 riser
                # 上把身体带滑(冒烟 run bhlrb5322 实证:暂停旋转中
                # z 4.85→4.12 坠落);yaw 修正交给后退/ramp(带 wz)。
                self.cmd.publish(Twist())
                self.hold_rl()
                self._record_trace()
                return
        if self.flight_ramp_until is not None:
            if now >= self.flight_ramp_until:
                self.flight_ramp_until=None
            else:
                # 与 _flight_control 同构:沿当前航向的纯轴 ramp 命令
                _vx, _vy, wz, _ce, _he=self._flight_control(which)
                body_axis=(-self.ramp_speed if self.backward_mode
                           else self.ramp_speed)
                vx=body_axis*math.cos(self.truth_pose[3])
                vy=body_axis*math.sin(self.truth_pose[3])
                self._publish_truth_world_command(vx, vy, wz)
                self._record_trace()
                return
        if self.flight_yaw_recover_until is not None:
            _vx, _vy, _wz, _ce, he=self._flight_control(which)
            if now >= self.flight_yaw_recover_until or \
                    abs(he) < self.yaw_recover_tolerance:
                self.flight_yaw_recover_until=None
                # 回正成功后重置侧滑守卫计数:卡死循环里 heading err
                # 反复超 0.45 会累积 attempts 导致 ALIGNMENT_LOST 误杀
                self.side_slip_attempts=0
                self.side_slip_pause_until=None
                self.flight_backoff_until=(
                    now+self.backoff_distance/self.backoff_speed)
                rospy.logwarn('Descent yaw recovery complete; backing off.')
            else:
                # 原地纯转体(不发平移),把 yaw 压回纯下行朝向。
                # 卡死时 yaw 斜站台阶冻结(flight_b 漂 1.89 / flight_a
                # 漂 -1.12),边走边转的执行力差,先静止回正再后退。
                yaw_rate=math.copysign(
                    min(self.landing_turn_speed,
                        abs(self.heading_gain*he)), he)
                self._publish_truth_world_command(0.0, 0.0, yaw_rate)
                self._record_trace()
                return
        if self.flight_backoff_until is not None:
            if now >= self.flight_backoff_until:
                self.flight_backoff_until=None
                self.flight_ramp_until=now+self.ramp_seconds
                # 2026-08-15 v2:后退(带 wz 修正)是一次主动恢复,重置
                # 侧滑守卫计数 —— v1 的 yaw-recovery 曾负责此重置,删除
                # 后 attempts 只涨不清,中心漂移 + 守卫 3 次即误杀
                # ALIGNMENT_LOST(冒烟 run: center err 0.32m)。
                self.side_slip_attempts=0
                self.side_slip_pause_until=None
                # 后退本身使 z 回升(退 2-3 级台阶可达 0.5-0.8 m):
                # GAIN_UP 守卫从后退结束后的当前 z 重新武装,否则恢复
                # 动作刚结束就触发回升高守卫(冒烟 R12 实锤)。
                self.flight_descent_min_z=self.truth_pose[2]
                rospy.logwarn('Descent backoff complete; ramp resume.')
            else:
                # 2026-08-15 flight_a 卡死修复:旧实现发布固定 world 轴
                # (0,±backoff),yaw 漂移时投影成 body 横向分量
                # (0.057→0.130,随漂移增大);stair 策略拒绝带横向的
                # 命令 —— RL 日志在 sim 213.4 收到 body=(-0.282,0.103)
                # 后完全停止,机器人冻结。改为与前进/ramp 同构:
                # 沿当前航向反方向纯轴后退,body 恒为 ±backoff 纯轴。
                body_axis=(self.backoff_speed if self.backward_mode
                           else -self.backoff_speed)
                vx=body_axis*math.cos(self.truth_pose[3])
                vy=body_axis*math.sin(self.truth_pose[3])
                # 后退期间输出更强的 yaw 伺服(backoff_yaw_rate 上限,
                # 高于飞行中 max_yaw_rate):stall 前的漂移边走边回正,
                # 避免恢复后斜行再卡;不原地转体(台阶边缘旋转会带滑,
                # run bhlrb5322 实证坠落)。
                _vx, _vy, _wz, _ce, he=self._flight_control(which)
                yaw_rate=math.copysign(
                    min(self.backoff_yaw_rate,
                        abs(self.heading_gain*he)), he)
                self._publish_truth_world_command(vx, vy, yaw_rate)
                self._record_trace()
                return
        # ---- 碎步停顿:每下 step_pause_interval_m 停顿片刻再继续 ----
        if self.step_pause_seconds > 0.0 and \
                drop >= self.stall_arm_drop and \
                (self.last_step_pause_drop is None or
                 drop - self.last_step_pause_drop >=
                 self.step_pause_interval_m):
            self.last_step_pause_drop=drop
            self.flight_pause_until=now+self.step_pause_seconds
            # 重置 stall 窗口:pause 期间 z 冻结是主动行为,不得计为卡死
            self.flight_stall_since=now
            self.flight_stall_window_start_z=self.truth_pose[2]
            rospy.loginfo('Descent step pause: drop %.3f m, hold %.1f s.',
                          drop, self.step_pause_seconds)
            self.cmd.publish(Twist())
            self.hold_rl()
            self._record_trace(force=True)
            return
        # stall:z 冻结速率检测(上梯同款),必须已下第一步才武装。
        # 武装条件 drop>=stall_arm_drop:平坦 landing 上走向楼梯时 z 不变,
        # 若不武装会把"还没走到楼梯"误判为卡死(backward-0.20 冒烟:
        # 20 次 stall 原地循环,SEGMENT_TIMEOUT)。
        # 检测条件:武装后持续 stall_seconds,期间 z 下降不足
        # stall_z_progress 即视为卡死。总 drop 已过 0.15m 后仍能检测
        # 中途卡死(旧 total-drop 设计一旦 drop>=0.15 就永久失效)。
        # 2026-08-15 修复:yaw 回正/backoff/ramp 等恢复子状态期间暂停
        # stall 检测 —— 否则回正 4s 内 z 冻结恰好满足 stall 窗口,
        # 不断重置 backoff 形成死循环(冒烟实测 stall #20-#23 原地循环)。
        if (self.flight_pause_until is not None or
                self.flight_yaw_recover_until is not None or
                self.flight_backoff_until is not None or
                self.flight_ramp_until is not None):
            self.flight_stall_since=None
            self.flight_stall_window_start_z=None
            self.flight_stall_window_start_y=None
        # fixV4:武装条件 = z 已起步 或 沿下梯方向已前进 stall_y_arm
        # (段2 FLIGHT_B 从 F2 landing 起步即卡时 drop≈0 永不武装)。
        if (drop >= self.stall_arm_drop or
                self._flight_y_advance() >= self.stall_y_arm) and \
                elapsed >= self.stall_min_elapsed:
            if self.flight_stall_since is None:
                self.flight_stall_since=now
                self.flight_stall_window_start_z=self.truth_pose[2]
                self.flight_stall_window_start_y=self.truth_pose[1]
            if now-self.flight_stall_since >= self.stall_seconds:
                # Snapshot callback-owned window state before computing.  A
                # concurrent fall/reset edge is allowed to clear either
                # anchor; in that case restart the bounded stall window and
                # defer judgement to the next tick.
                window_z=self.flight_stall_window_start_z
                window_y=self.flight_stall_window_start_y
                if window_z is None or window_y is None:
                    self.flight_stall_since=now
                    self.flight_stall_window_start_z=self.truth_pose[2]
                    self.flight_stall_window_start_y=self.truth_pose[1]
                    rospy.logwarn_throttle(
                        2.0, 'Descent stall window reset concurrently; '
                        're-arming from current truth pose.')
                    return
                z_drop=window_z-self.truth_pose[2]
                y_adv=self._flight_y_advance_from(window_y)
                self.flight_stall_since=now
                self.flight_stall_window_start_z=self.truth_pose[2]
                self.flight_stall_window_start_y=self.truth_pose[1]
                if z_drop < self.stall_z_progress and \
                        y_adv < self.stall_y_progress:
                    self.flight_stall_recoveries+=1
                    # A body that drops only one riser and then freezes at
                    # the upper lip is different from an ordinary mid-flight
                    # stall.  firstg4gatefix stalled at 0.195 m for 30 generic
                    # recoveries.  After two measured freeze windows, place it
                    # once on the centre of the third real tread using parsed
                    # Gazebo geometry.  This remains bounded and all remaining
                    # descent/completion gates stay physical.
                    partial_lip_stall = bool(
                        which == 'b' and
                        self.flight_stall_recoveries >=
                        self.truth_partial_lip_stall_trigger and
                        0.04 <= peak_drop <= self.truth_partial_lip_max_drop)
                    if partial_lip_stall:
                        if (self.truth_no_drop_corrections <
                                self.truth_no_drop_max_corrections and
                                self._correct_no_drop_stair_seam(
                                    which, tread_index=3,
                                    reason='partial_drop_top_lip_stall')):
                            return
                        self._terminal(
                            'STAIR_DESCENT_PARTIAL_LIP_CORRECTION_FAILED',
                            'stair_descent_partial_lip_correction_failed',
                            error=True)
                        return
                    # fixV(2026-08-19):stall 上限 —— RUN1 实证 FLIGHT_B
                    # 段 2 从侧滑落点(x=-1.90,tread 东边缘外)起步推不动,
                    # 83 次 backoff/ramp 循环 573s 才 SEGMENT_TIMEOUT。
                    # 卡死结构(起步位置/姿态错误)时 recovery 无限循环无
                    # 进展,上限快速终止暴露失败轮次。
                    if self.flight_stall_recoveries >= self.flight_max_stall_recoveries:
                        rospy.logwarn(
                            'Descent stall #%d: exceeded max recoveries %d '
                            'at z=%.3f; aborting descent.',
                            self.flight_stall_recoveries,
                            self.flight_max_stall_recoveries,
                            self.truth_pose[2])
                        self._terminal('STAIR_DESCENT_FLIGHT_STALL_ABORT',
                                       'stair_descent_flight_stall_abort',
                                       error=True)
                        return
                    if self.flight_stall_recoveries >= 2:
                        # 2026-08-15 v2:不再原地纯转体(yaw-recovery 旋转
                        # 4s 在台阶边缘带滑坠落,run bhlrb5322 实证 yaw
                        # 2.05→2.25 同时 z 4.85→4.12)。直接后退,后退
                        # 期间 wz(backoff_yaw_rate 上限)边走边修正 yaw;
                        # 后退距离随 stall 次数升级:卡在同一 riser 时
                        # 固定 0.35m 只退到上一步,重新起步又卡同一级
                        # (stall #1-#9 原地循环实证),升级后退更远给
                        # 更长助跑(0.75m ≈ 退 2~3 级)。
                        _vx, _vy, _wz, _ce, _he=self._flight_control(which)
                        distance=min(
                            self.backoff_distance +
                            self.backoff_escalation_m *
                            (self.flight_stall_recoveries-2),
                            self.backoff_max_distance)
                        self.flight_backoff_until=(
                            now+distance/self.backoff_speed)
                        rospy.logwarn(
                            'Descent stall #%d: backing off %.2f m down '
                            'the flight for a fresh run-up (z drop %.3f '
                            'm in %.1f s).', self.flight_stall_recoveries,
                            distance, z_drop, self.stall_seconds)
                    else:
                        self.flight_pause_until=now+self.stall_pause_seconds
                        rospy.logwarn(
                            'Descent stall #%d: pausing %.1f s (z drop %.3f '
                            'm in %.1f s).', self.flight_stall_recoveries,
                            self.stall_pause_seconds, z_drop,
                            self.stall_seconds)
                    self._record_trace(force=True)
                    return
        else:
            self.flight_stall_since=None
            self.flight_stall_window_start_z=None
            self.flight_stall_window_start_y=None
        # ---- 侧滑守卫(下梯中段后中心/航向偏差过大) ----
        # 2026-08-15 v2 重写计数:暂停结束时才计数 + 修正窗口。
        # v1 触发即计数,每次 pause 结束立刻再触发(间隔 2s),0.3m
        # 稳态中心漂移 3 次即误杀 ALIGNMENT_LOST(冒烟 run 实测
        # center err 0.303→0.329 连杀)。0.3m 在 1.43m 宽楼梯上安全,
        # 守卫应只拦危险漂移;计数前给横向引导(steer)修正时间。
        if drop >= self.side_slip_min_drop:
            _vx, _vy, _wz, center_error, heading_error=self._flight_control(which)
            if (abs(center_error) > self.max_center_error or
                    abs(heading_error) > self.max_heading_error):
                if self.side_slip_pause_until is None:
                    if (self.side_slip_next_check is None or
                            now >= self.side_slip_next_check):
                        self.side_slip_pause_until=(
                            now+self.side_slip_pause_seconds)
                        rospy.logwarn(
                            'Descent side-slip guard: center err %.3f m, '
                            'heading err %.3f rad; pausing %.1f s.',
                            center_error, heading_error,
                            self.side_slip_pause_seconds)
                elif now >= self.side_slip_pause_until:
                    self.side_slip_pause_until=None
                    # pause 结束时才计数:期间 error 可能已回落(静止
                    # 后姿态修正);回落则不浪费 attempts。
                    _vx2, _vy2, _wz2, ce2, he2=self._flight_control(which)
                    if (abs(ce2) > self.max_center_error or
                            abs(he2) > self.max_heading_error):
                        self.side_slip_attempts+=1
                        rospy.logwarn(
                            'Descent side-slip attempt %d/%d (center '
                            'err %.3f m, heading err %.3f rad).',
                            self.side_slip_attempts,
                            self.side_slip_max_attempts, ce2, he2)
                        if self.side_slip_attempts > self.side_slip_max_attempts:
                            self._terminal('STAIR_DESCENT_ALIGNMENT_LOST',
                                           'stair_descent_alignment_lost',
                                           error=True)
                            return
                    # 修正窗口:pause 后给横向引导/后退足够时间拉回,
                    # 窗口内不再触发(间隔由 pause+窗口构成,远大于 v1
                    # 的 2s 连杀)
                    self.side_slip_next_check=(
                        now+self.side_slip_guard_window_seconds)
                else:
                    self.cmd.publish(Twist())
                    self.hold_rl()
                    self._record_trace()
                    return
        # ---- 段超时 ----
        if self._segment_elapsed() >= self.descent_segment_timeout:
            self._terminal('STAIR_DESCENT_SEGMENT_TIMEOUT',
                           'stair_descent_segment_timeout', error=True)
            return
        vx, vy, wz, _ce, _he=self._flight_control(which)
        self._publish_truth_world_command(vx, vy, wz)
        self._record_trace()

    # ------------------------------------------------------------------ #
    # 中间平台转体(STAIR_DESCENT_TURN)
    # ------------------------------------------------------------------ #
    def _landing_turn_tick(self):
        if self.truth_pose is None:
            return
        now=time.monotonic()
        # ---- R36 落步稳定:进入 TURN 后先零命令 settle 数秒,等 FLIGHT_B
        #      落地后的 z/姿态波动稳定再启动转体步态(起步即转会反复滑落)。
        if self.landing_settle_remaining > 0.0:
            self.cmd.publish(Twist())
            self.hold_rl()
            self.landing_settle_remaining-=0.02
            self._record_trace()
            return
        # ---- 滑落恢复(TURN 期间 z 骤降 = 从台阶边缘滑到 landing 平面,
        #      转体步态打滑 yaw 无法转动;重新落步 + 回退台阶后重试) ----
        if self.landing_slip_active:
            if now < self.landing_slip_until:
                # 阶段1:零命令重新落步(FLIGHT_B stall pause 同款)
                self.cmd.publish(Twist())
                self.hold_rl()
                self._record_trace()
                return
            if self.landing_slip_backoff_until is None:
                self.landing_slip_backoff_until=(now +
                    self.landing_slip_backoff_distance /
                    max(0.05, self.landing_slip_backoff_speed))
                rospy.logwarn(
                    'Descent landing slip recovery: backing east onto the '
                    'stair (%.2f m) before retrying the turn.',
                    self.landing_slip_backoff_distance)
            if now < self.landing_slip_backoff_until:
                # 阶段2:world +x 回退(本场景 TURN 横移方向恒为朝 flight_a
                # 的 -x,滑落即落在西侧,回退 +x 回台阶;纯 world 命令避免
                # body 投影随 yaw 翻转)。R31:滑落根因是 y 滑出安全带
                # (4.70<4.77 北缘,z 掉 0.24m),backoff 同时按 y 偏差回推
                # y 回 turn_y_keep。
                y_back=0.0
                if self.truth_pose[1] < self.turn_y_safe_low:
                    y_back=min(self.landing_recenter_speed,
                               max(0.15, (self.turn_y_keep-self.truth_pose[1])
                                   *1.0))
                self._publish_truth_world_command(
                    self.landing_slip_backoff_speed, y_back, 0.0)
                self._record_trace()
                return
            # 恢复完成:重置转体基准与计时再试
            self.landing_slip_active=False
            self.landing_slip_backoff_until=None
            self.landing_z0=self.truth_pose[2]
            self.landing_started_at=now
            self.landing_recovery_active=False
            self.landing_yaw_stall_accum=0.0
            self.landing_last_yaw=None
            self.landing_last_yaw_at=None
            self.landing_turn_direction=None
            rospy.logwarn('Descent landing re-planted at z=%.2f m; '
                          'retrying turn.', self.truth_pose[2])
        if self.landing_z0 is not None:
            z_drop=self.landing_z0-self.truth_pose[2]
            if (z_drop > self.landing_slip_drop and
                    self.landing_slip_recoveries <
                    self.landing_slip_max_attempts):
                self.landing_slip_recoveries+=1
                self.landing_slip_active=True
                self.landing_slip_until=now+self.landing_slip_pause_seconds
                self.landing_slip_backoff_until=None
                rospy.logwarn(
                    'Descent landing slip: z dropped %.2f m; re-planting '
                    'gait (recovery %d/%d).', z_drop,
                    self.landing_slip_recoveries,
                    self.landing_slip_max_attempts)
                self._record_trace()
                return
        landing_elapsed=(now-self.landing_started_at
                         if self.landing_started_at is not None else 0.0)
        target_heading=(self._flight_heading(self.truth_flight_a_heading)
                        if self.truth_flight_a_heading is not None
                        else -math.pi/2.0)
        target_x=self._flight_a_center_x()
        raw_error=target_heading-self.truth_pose[3]
        error=math.atan2(math.sin(raw_error), math.cos(raw_error))
        # R36: 起始 yaw 与目标差 ~π(直径两端等距)时 atan2 error 符号在
        # ±π 边界随 yaw 微动抖动 → wz 在 ±0.4 横跳 → 转体步态反复起步
        # 滑落。方向锁存:首次进入固定转体方向,再用无界误差(与锁存方向
        # 一致的 0→π 单调区间)驱动,误差单调降到 0,命令不再翻号。
        if self.landing_turn_direction is None:
            if abs(error) >= 2.0:
                self.landing_turn_direction=1
            else:
                self.landing_turn_direction=1 if error >= 0.0 else -1
        unbounded=math.fmod(raw_error, 2.0*math.pi)
        if self.landing_turn_direction > 0 and unbounded <= 0.0:
            unbounded+=2.0*math.pi
        elif self.landing_turn_direction < 0 and unbounded >= 0.0:
            unbounded-=2.0*math.pi
        # The half-turn direction latch is only needed around the initial
        # +/-pi ambiguity.  Once the target heading is nearby, position
        # recentering may still be unfinished.  Keeping the positive
        # unbounded error in that state converts a tiny negative yaw error
        # into almost +2pi and commands a needless full extra revolution
        # (minimal-stack run11: 16.9 s instead of the normal 6-7 s turn).
        # Hold/correct heading by the shortest signed error while x settles.
        direction_release_error=max(0.60,
                                    2.0*self.landing_heading_tolerance)
        turn_error=(error if abs(error) <= direction_release_error
                    else unbounded)
        position_error=target_x-self.truth_pose[0]
        self.landing_heading_error=error
        self.landing_position_error=position_error
        if (abs(error) <= self.landing_heading_tolerance and
                abs(position_error) <= self.landing_position_tolerance):
            self.landing_recovery_active=False
            self._begin_flight('a')
            rospy.loginfo('Descent landing aligned: x err=%.3f m, '
                          'heading err=%.3f rad; starting flight A.',
                          position_error, error)
            return
        # ---- yaw 冻结检测(转体命令持续但 yaw 几乎不动 = 转体步态在平台
        #      边缘打滑;R13 实证:最后 9° 冻结 15s,z 稳定无骤降,recovery
        #      只提 yaw rate 下限物理上仍转不动 → 54s TURN_TIMEOUT。
        #      与 z 骤降同款恢复:零命令落步 → +x 回退 → 重置重试) ----
        if abs(error) > self.landing_heading_tolerance:
            dt=(now-self.landing_last_yaw_at
                if self.landing_last_yaw_at is not None else 0.0)
            # R31: 50Hz tick(dt≈0.02s)使 0.05 下界永不满足,stall 检测
            # 完全失效(R30 实证:align 侧移冻结 37s 无恢复→超时)。
            # 去掉下界,dt>0 即累积。
            if (self.landing_last_yaw is not None and 0.0 < dt < 2.0 and
                    abs(self.truth_pose[3]-self.landing_last_yaw) <
                    self.landing_yaw_stall_delta_rad):
                self.landing_yaw_stall_accum+=dt
            else:
                self.landing_yaw_stall_accum=0.0
            self.landing_last_yaw=self.truth_pose[3]
            self.landing_last_yaw_at=now
            if (self.landing_yaw_stall_accum >=
                    self.landing_yaw_stall_seconds and
                    self.landing_slip_recoveries <
                    self.landing_slip_max_attempts):
                self.landing_slip_recoveries+=1
                self.landing_slip_active=True
                self.landing_slip_until=now+self.landing_slip_pause_seconds
                self.landing_slip_backoff_until=None
                self.landing_yaw_stall_accum=0.0
                rospy.logwarn(
                    'Descent landing yaw stall: heading err %.3f rad stuck '
                    '%.1f s; re-planting gait (recovery %d/%d).',
                    error, self.landing_yaw_stall_seconds,
                    self.landing_slip_recoveries,
                    self.landing_slip_max_attempts)
                self._record_trace()
                return
        else:
            self.landing_yaw_stall_accum=0.0
            self.landing_last_yaw=None
            self.landing_last_yaw_at=None
        # 超时 → 有界恢复重试,耗尽后终态
        timeout=(self.landing_recovery_timeout
                 if self.landing_recovery_active else self.landing_timeout)
        if landing_elapsed >= timeout:
            if (self.landing_recovery_attempts <
                    self.landing_recovery_max_attempts and
                    abs(position_error) <=
                    self.landing_recovery_max_position_error and
                    abs(error) <= self.landing_recovery_max_heading_error):
                self.landing_recovery_attempts+=1
                self.landing_recovery_active=True
                self.landing_started_at=time.monotonic()
                self.landing_z0=self.truth_pose[2]
                self.landing_slip_active=False
                self.landing_slip_backoff_until=None
                self.landing_yaw_stall_accum=0.0
                self.landing_last_yaw=None
                self.landing_last_yaw_at=None
                self.landing_turn_direction=None
                rospy.logwarn(
                    'Descent landing alignment retry %d/%d: position '
                    'err=%.3f m, heading err=%.3f rad.',
                    self.landing_recovery_attempts,
                    self.landing_recovery_max_attempts,
                    position_error, error)
            else:
                self._terminal('STAIR_DESCENT_TURN_TIMEOUT',
                               'stair_descent_turn_timeout', error=True)
                return
        cross_speed=self._landing_cross_rate(position_error)
        yaw_rate=self._landing_turn_rate(turn_error)
        # R36 慢启动:落步稳定后头 2s 转体速度渐进,避免大步态起步瞬间
        # 滑落;起点 0.24 = 策略 yaw 死区上缘(0.10 起步转不动);
        # 2s 内渐进到满速 landing_turn_speed,2s 后恢复满速。
        ramp_since=landing_elapsed-self.landing_settle_seconds
        if 0.0 <= ramp_since < 2.0:
            cap=(0.24 + (self.landing_turn_speed-0.24)*ramp_since/2.0)
            yaw_rate=max(-cap, min(cap, yaw_rate))
        # 世界 +y 前推,仅在转体步态把机器人漂到 y < hold(北,朝 flight_a
        # 台顶)时轻微回推,把 y 保持在上梯已验证的安全带 4.77-4.85。
        # 旧实现 (limit-y)*2 起始即满速(0.06-0.12 m/s),2-3s 就把机器人
        # 推到 y=5.02-5.20 越过平台南缘,转体步态物理死锁(round 2/5 实证,
        # 上梯同平台转体 vy=0 成功)。gain 1.0 + 0.03 上限 ≈ 轻微保持。
        forward=min(self.turn_forward_speed,
                    max(0.0, (self.turn_forward_y_limit-self.truth_pose[1])
                        *1.0))
        # R31:转体期间 y 滑出安全带(4.70<4.77 北缘)→ 脚滑出平台 → z 掉
        # 0.24m → 转体步态死锁,恢复 4/4 耗尽仍 TURN_TIMEOUT。y<下缘时
        # 暂停转体(预防优先于恢复),0.3 m/s 起步把 y 推回 turn_y_keep。
        if self.truth_pose[1] < self.turn_y_safe_low:
            yaw_rate=0.0
            forward=min(0.6, max(0.3,
                                 (self.turn_y_keep-self.truth_pose[1])*2.0))
            self._publish_truth_world_command(cross_speed, forward, yaw_rate)
            self._record_trace()
            return
        self._publish_truth_world_command(cross_speed, forward, yaw_rate)
        self._record_trace()

    def _landing_turn_rate(self, heading_error):
        yaw_rate=max(-self.landing_turn_speed,
                     min(self.landing_turn_speed, 1.10*heading_error))
        minimum=min(self.landing_turn_speed,
                    max(0.0, self.landing_minimum_yaw_rate))
        if self.landing_recovery_active:
            minimum=min(self.landing_turn_speed,
                        max(minimum, self.landing_recovery_minimum_yaw_rate))
        if (abs(heading_error) > self.landing_heading_tolerance and
                abs(yaw_rate) < minimum):
            yaw_rate=math.copysign(minimum, heading_error)
        return yaw_rate

    def _landing_cross_rate(self, position_error):
        if abs(position_error) <= self.landing_position_tolerance:
            return 0.0
        maximum=max(0.0, self.landing_recenter_speed)
        # The stair policy has the same translational dead zone here as in
        # PRE_ALIGN/EAST_ALIGN.  run35 finished the landing turn with only
        # 0.049 rad heading error, but an x error of 0.203 m sat 3 mm outside
        # the configured 0.20 m gate.  The old 0.183 m/s correction never
        # moved the feet and consumed the whole 45 s TURN watchdog.  Reuse
        # the explicit minimum effective x speed until the physical position
        # gate is met; the command remains capped by landing_recenter_speed.
        minimum=min(maximum, max(0.0, self.landing_minimum_x_speed))
        requested=self.landing_position_gain*position_error
        magnitude=min(maximum, max(minimum, abs(requested)))
        return math.copysign(magnitude, position_error)

    # ------------------------------------------------------------------ #
    # 段间转体(F2 landing 上 180° 掉头)
    # ------------------------------------------------------------------ #
    def _segment_turn_strong_rate(self, turn_error):
        """强转身 yaw 命令:2.0×turn_error(上梯 landing turn 同款增益,
        R36 后传入单调无界误差,err_pos 正转特判已由方向锁存替代),
        下限 0.24 冲过策略低速率 yaw 死区。"""
        yaw_rate=max(-self.segment_turn_yaw_rate,
                     min(self.segment_turn_yaw_rate, 2.0*turn_error))
        if abs(yaw_rate) < self.landing_minimum_yaw_rate:
            yaw_rate=math.copysign(self.landing_minimum_yaw_rate, yaw_rate)
        return yaw_rate

    def _begin_segment_turn_fixed_stand_recovery(self, reason):
        """Physically level the dog once, then reload the stair gait.

        This recovery is only valid on the verified inter-floor landing.  It
        deliberately does not call ``/gazebo/set_model_state``: strict room
        evidence may be incomplete, but the trip home must still be physical.
        """
        if (not self.segment_turn_fixed_stand_enabled or
                self.segment_turn_fixed_stand_attempts >=
                self.segment_turn_fixed_stand_max_attempts):
            return False
        self.segment_turn_fixed_stand_attempts+=1
        now=time.monotonic()
        self.segment_turn_fixed_stand_started_at=now
        self.segment_turn_fixed_stand_last_request=None
        self.phase='STAIR_DESCENT_SEGMENT_FIXED_STAND_RECOVERY'
        self.state.publish(String(data=self.phase))
        self.cmd.publish(Twist())
        rospy.logwarn(
            'Descent segment turn remained frozen on the flat landing; '
            'starting bounded physical FixedStand recovery %d/%d (%s).',
            self.segment_turn_fixed_stand_attempts,
            self.segment_turn_fixed_stand_max_attempts, reason)
        self._record_trace(force=True)
        return True

    def _segment_turn_fixed_stand_tick(self):
        now=time.monotonic()
        self.cmd.publish(Twist())
        # FixedStand is edge-triggered.  Reassert slowly in case the first Joy
        # packet coincides with a controller transition, without flooding the
        # FSM at the 50 Hz descent tick rate.
        if (self.segment_turn_fixed_stand_last_request is None or
                now-self.segment_turn_fixed_stand_last_request >= 0.75):
            stand=Joy()
            stand.header.stamp=rospy.Time.now()
            stand.axes=[0.0]*8
            stand.buttons=[0]*12
            stand.buttons[1]=1  # L2_A / FixedStand
            self.joy.publish(stand)
            self.segment_turn_fixed_stand_last_request=now
        if (now-self.segment_turn_fixed_stand_started_at <
                self.segment_turn_fixed_stand_seconds):
            self._record_trace()
            return
        self.active_policy=self.policy
        self.policy_loaded=False
        self.phase='STAIR_DESCENT_SEGMENT_POLICY_LOADING'
        self.segment_policy_loading_started_at=now
        self.segment_policy_loading_last_request=now
        self.segment_policy_loading_request_count=1
        self.state.publish(String(data=self.phase))
        self.pub.publish(String(data=self.policy))
        # FixedStand changes the controller mode.  Reassert the RL mode in
        # the same transaction; run25 published the policy request alone and
        # then waited forever because no policy_reloaded acknowledgement was
        # produced while the FSM remained in FixedStand.
        self.hold_rl()
        rospy.logwarn('Segment FixedStand recovery complete; reloading stair '
                      'policy before retrying the landing turn.')
        self._record_trace(force=True)

    def _segment_policy_loading_tick(self):
        """Bound policy reload after FixedStand and preserve the trip home."""
        now=time.monotonic()
        self.cmd.publish(Twist())
        self.hold_rl()
        # If the dog has already physically reached the first-floor support
        # while this acknowledgement was missing, do not keep commanding a
        # stale F2 turn forever.  Continue with the plane-policy home leg but
        # label the stair descent degraded; this is continuity, not strict
        # physical segment-completion credit.
        attitude_ok=bool(
            self.truth_attitude is None or
            (abs(self.truth_attitude[0]) <= self.home_maximum_tilt and
             abs(self.truth_attitude[1]) <= self.home_maximum_tilt))
        if (self.truth_pose is not None and attitude_ok and
                self.home_minimum_upright_z <= self.truth_pose[2] <=
                self.segment_policy_lower_floor_z):
            self.first_floor_landing_ros_time=rospy.Time.now().to_sec()
            self._publish_milestone(
                'F1_LANDING_REACHED_DEGRADED_POLICY_ACK_TIMEOUT')
            self.first_floor_state_pub.publish(String(
                data='FIRST_FLOOR_LANDING_REACHED_DEGRADED'))
            rospy.logerr(
                'Segment policy acknowledgement was lost, but truth confirms '
                'an upright first-floor landing at z=%.3f; continuing the '
                'home leg without strict segment-2 completion credit.',
                self.truth_pose[2])
            self._begin_home_return()
            self._record_trace(force=True)
            return
        if self.policy_loaded:
            self.phase='STAIR_DESCENT_SEGMENT_POLICY_WARMUP'
            self.state.publish(String(data=self.phase))
            self.policy_warmup_until=now+self.policy_warmup_seconds
            self._record_trace(force=True)
            return
        if (self.segment_policy_loading_last_request is None or
                (now-self.segment_policy_loading_last_request >=
                 self.segment_policy_loading_retry_period and
                 self.segment_policy_loading_request_count <
                 self.segment_policy_loading_max_requests)):
            self.pub.publish(String(data=self.policy))
            self.segment_policy_loading_last_request=now
            self.segment_policy_loading_request_count+=1
            rospy.logwarn(
                'Reasserting stair policy after FixedStand (%d/%d).',
                self.segment_policy_loading_request_count,
                self.segment_policy_loading_max_requests)
        if (self.segment_policy_loading_started_at is not None and
                now-self.segment_policy_loading_started_at >=
                self.segment_policy_loading_timeout):
            self._terminal(
                'STAIR_DESCENT_SEGMENT_POLICY_LOADING_TIMEOUT',
                'stair_descent_segment_policy_loading_timeout', error=True)
            return
        self._record_trace()

    def _resume_segment_turn_after_fixed_stand(self):
        now=time.monotonic()
        self.phase='STAIR_DESCENT_SEGMENT_TURN'
        self.state.publish(String(data=self.phase))
        self.segment_turn_started_at=now
        self.segment_turn_stage='align'
        self.segment_turn_direction=None
        self.segment_turn_turn_started_at=None
        self.landing_settle_remaining=0.5
        self.landing_z0=self.truth_pose[2] if self.truth_pose else None
        self.landing_yaw_stall_accum=0.0
        self.landing_last_yaw=None
        self.landing_last_yaw_at=None
        self.landing_slip_active=False
        self.landing_slip_until=None
        self.landing_slip_backoff_until=None
        rospy.logwarn('Stair policy restored after FixedStand; retrying '
                      'inter-floor turn with a fresh bounded watchdog.')
        self._record_trace(force=True)

    def _segment_turn_tick(self):
        if self.truth_pose is None:
            return
        now=time.monotonic()
        if now-self.segment_turn_started_at >= self.segment_turn_timeout:
            if self._begin_segment_turn_fixed_stand_recovery('turn_timeout'):
                return
            self._terminal('STAIR_DESCENT_SEGMENT_TURN_TIMEOUT',
                           'stair_descent_segment_turn_timeout', error=True)
            return
        # ---- R37 落步稳定:FLIGHT_A 末端着地后先零命令 settle,等步态
        #      站稳再掉头(R17 实证着地打滑导致 align/转体冻结) ----
        if self.landing_settle_remaining > 0.0:
            self.cmd.publish(Twist())
            self.hold_rl()
            self.landing_settle_remaining-=0.02
            self._record_trace()
            return
        # ---- 滑落/冻结恢复(landing TURN 同款:零命令落步 → world +x
        #      回退 → 重置重试;R17 实证:flight_a 末端跌落着地后步态
        #      打滑,align/转体全部冻结 45s → SEGMENT_TURN_TIMEOUT) ----
        if self.landing_slip_active:
            if now < self.landing_slip_until:
                self.cmd.publish(Twist())
                self.hold_rl()
                self._record_trace()
                return
            if self.landing_slip_backoff_until is None:
                self.landing_slip_backoff_until=(now +
                    self.landing_slip_backoff_distance /
                    max(0.05, self.landing_slip_backoff_speed))
                rospy.logwarn(
                    'Descent segment-turn slip recovery: backing east '
                    '(%.2f m) before retrying.',
                    self.landing_slip_backoff_distance)
            if now < self.landing_slip_backoff_until:
                self._publish_truth_world_command(
                    self.landing_slip_backoff_speed, 0.0, 0.0)
                self._record_trace()
                return
            self.landing_slip_active=False
            self.landing_slip_backoff_until=None
            self.segment_turn_started_at=now
            self.segment_turn_stage='align'
            self.landing_z0=self.truth_pose[2]
            self.landing_yaw_stall_accum=0.0
            self.landing_last_yaw=None
            self.landing_last_yaw_at=None
            self.segment_turn_direction=None
            rospy.logwarn('Descent segment-turn re-planted; retrying.')
        landing=self.truth_landing_center or (-3.25, 1.55)
        center_x=self.segment_turn_x_keep
        target=(self._flight_heading(self.truth_flight_b_heading)
                if self.truth_flight_b_heading is not None
                else math.pi/2.0)
        x_err=center_x-self.truth_pose[0]
        # y 目标 = 掉头安全带(landing 平坦区北半),NOT landing 南缘:
        # R20/R21 实证 y<2.05 已是 flight_b 阶梯 tread(z 低 0.45m),
        # align/turn 把 y 拉向 1.55/1.75 会在阶梯上横移转动 → 打滑跌落/
        # 侧翻(R20 东滑卡死,R21 roll≈2.65rad 翻倒)。
        y_err=self.segment_turn_y_keep-self.truth_pose[1]
        raw_error=target-self.truth_pose[3]
        error=math.atan2(math.sin(raw_error), math.cos(raw_error))
        # R36 同款方向锁存:π 边界(FLIGHT_A 后 yaw≈-π/2 对目标 +π/2,
        # 差≈π)处 atan2 符号随 yaw 微动抖动 → wz 横跳 → 步态反复起步
        # 打滑。锁存方向后无界误差单调降到 0。
        # R37 曾用 |error|≥2 强制正转,但 full17 实证:yaw=-1.7 对目标
        # +1.57 时正转 188° 前半段 lat 投影纯东移 ~0.8m,东滑出 F2
        # landing 掉进 flight_b 楼梯(z 3.04→2.62)打滑冻结 70s。
        # R38(fixO)锁存 sign(error) 仍不可靠:full3 实证 settle 3s 内
        # yaw -1.589→-1.480(微过 -π/2),sin(raw)>0 → error=+3.05 →
        # 锁正转 172°(经 0)。正转 lat 投影 y=lat·cos 全程≥0 北漂
        # +0.27m 把 y 推到 2.37(flight_b tread 上)→ EAST_ALIGN 侧移
        # 跨 tread 边缘打滑 → EAST_ALIGN_TIMEOUT。
        # 修复 P:短路径≈π 时(误差过半圈)两条路径几乎等长,但 lat
        # 投影 y 方向相反 —— 正转北漂(tread 方向,危险)、反转经 -π
        # 南漂(平台内,安全)。|error|≥π/2 无条件选反转;其余走
        # sign(error) 短路径。
        if self.segment_turn_direction is None:
            if abs(error) >= math.pi/2.0:
                self.segment_turn_direction=-1
            else:
                self.segment_turn_direction=1 if error >= 0.0 else -1
        unbounded=math.fmod(raw_error, 2.0*math.pi)
        if self.segment_turn_direction > 0 and unbounded <= 0.0:
            unbounded+=2.0*math.pi
        elif self.segment_turn_direction < 0 and unbounded >= 0.0:
            unbounded-=2.0*math.pi
        turn_error=unbounded
        # ---- 滑落触发:z 骤降(flight_a 末端跌落着地,着地瞬间姿态跳变
        #      步态打滑)——与 landing TURN 同款检测与恢复 ----
        if self.landing_z0 is not None:
            z_drop=self.landing_z0-self.truth_pose[2]
            if (z_drop > self.landing_slip_drop and
                    self.landing_slip_recoveries <
                    self.landing_slip_max_attempts):
                self.landing_slip_recoveries+=1
                self.landing_slip_active=True
                self.landing_slip_until=now+self.landing_slip_pause_seconds
                self.landing_slip_backoff_until=None
                rospy.logwarn(
                    'Descent segment-turn slip: z dropped %.2f m; '
                    're-planting gait (recovery %d/%d).', z_drop,
                    self.landing_slip_recoveries,
                    self.landing_slip_max_attempts)
                self._record_trace()
                return
        # 转身完成判定放宽到 0.20 rad:FLIGHT_B 的 heading-bias 伺服会
        # 修残差,而 settle 阶段 0.24 下限的 wz 会把 yaw 锁在 ±0.12 极限
        # 环内,严格 0.12 门可能永不触发。
        turn_done=abs(error) <= max(0.20, self.segment_turn_heading_tolerance)
        position_ok=(abs(x_err) <= self.segment_turn_position_tolerance and
                     self.segment_turn_y_low <= self.truth_pose[1] <=
                     self.segment_turn_y_high)
        # ---- yaw/位置冻结检测(命令持续但姿态几乎不动 = 步态打滑;与
        #      landing TURN 同款累积检测;align 阶段 wz=0 时 yaw 漂移
        #      缓慢同样累积 → 打滑恢复) ----
        if not (turn_done and position_ok):
            dt=(now-self.landing_last_yaw_at
                if self.landing_last_yaw_at is not None else 0.0)
            # R31: 50Hz tick(dt≈0.02s)使 0.05 下界永不满足,stall 检测
            # 完全失效(R30 实证:align 侧移冻结 37s 无恢复→超时)。
            # 去掉下界,dt>0 即累积。
            if (self.landing_last_yaw is not None and 0.0 < dt < 2.0 and
                    abs(self.truth_pose[3]-self.landing_last_yaw) <
                    self.landing_yaw_stall_delta_rad):
                self.landing_yaw_stall_accum+=dt
            else:
                self.landing_yaw_stall_accum=0.0
            self.landing_last_yaw=self.truth_pose[3]
            self.landing_last_yaw_at=now
            if (self.landing_yaw_stall_accum >=
                    self.landing_yaw_stall_seconds and
                    self.landing_slip_recoveries <
                    self.landing_slip_max_attempts):
                self.landing_slip_recoveries+=1
                self.landing_slip_active=True
                self.landing_slip_until=now+self.landing_slip_pause_seconds
                self.landing_slip_backoff_until=None
                self.landing_yaw_stall_accum=0.0
                rospy.logwarn(
                    'Descent segment-turn yaw stall: heading err %.3f rad '
                    'stuck %.1f s; re-planting gait (recovery %d/%d).',
                    error, self.landing_yaw_stall_seconds,
                    self.landing_slip_recoveries,
                    self.landing_slip_max_attempts)
                self._record_trace()
                return
            if (self.landing_yaw_stall_accum >=
                    self.landing_yaw_stall_seconds and
                    self.landing_slip_recoveries >=
                    self.landing_slip_max_attempts and
                    self._begin_segment_turn_fixed_stand_recovery(
                        'yaw_stall_replants_exhausted')):
                return
        else:
            self.landing_yaw_stall_accum=0.0
            self.landing_last_yaw=None
            self.landing_last_yaw_at=None
        if turn_done and position_ok:
            self.segment_turn_done=True
            self._publish_milestone('F2_SECOND_DESCENT_STAGED')
            self._begin_stand()
            rospy.loginfo('Segment turn complete; standing before descent '
                          'segment %d.', self.segment+1)
            self._record_trace()
            return
        if self.segment_turn_stage == 'align':
            # 阶段 1:横向对中到 flight_b 中心线 + 纵向对中到 y_keep
            # (掉头安全带中位)。PRE_ALIGN 同源 —— 平坦 landing 上纯 body
            # 侧移(世界 x/y 伺服),wz=0 避免转向耦合。R20 实证:align 只对
            # x 收敛(y=1.95 未到 y_keep 就切 turn),掉头在贴 riser
            # (y=2.05)处打滑,东滑进 flight_b 起点阶梯,z 3.06→2.61
            # 跌落卡死。y 必须收敛到 y_keep 附近才允许掉头。
            if (abs(x_err) <= self.segment_turn_position_tolerance and
                    self.segment_turn_y_low <= self.truth_pose[1] <=
                    self.segment_turn_y_high):
                self.segment_turn_stage='turn'
                self.segment_turn_turn_started_at=time.monotonic()
                rospy.loginfo('Segment turn aligned (x err=%.3f m, '
                              'y=%.3f in band [%.2f, %.2f]); '
                              'starting 180 deg turn.',
                              x_err, self.truth_pose[1],
                              self.segment_turn_y_low,
                              self.segment_turn_y_high)
            else:
                vx=max(-self.landing_recenter_speed,
                       min(self.landing_recenter_speed,
                           self.landing_position_gain*x_err))
                if abs(vx) < self.landing_minimum_x_speed:
                    vx=math.copysign(self.landing_minimum_x_speed, vx)
                vy=0.0
                if self.truth_pose[1] < self.segment_turn_y_low:
                    vy=max(-self.segment_turn_side_speed,
                           min(self.segment_turn_side_speed,
                               self.landing_position_gain*
                               (self.segment_turn_y_low-
                                self.truth_pose[1])))
                elif self.truth_pose[1] > self.segment_turn_y_high:
                    vy=max(-self.segment_turn_side_speed,
                           min(self.segment_turn_side_speed,
                               self.landing_position_gain*
                               (self.segment_turn_y_high-
                                self.truth_pose[1])))
                self._publish_truth_world_command(vx, vy, 0.0)
                self._record_trace()
                return
        if self.segment_turn_stage == 'turn':
            # 阶段 2:180° 掉头。stair 策略的转身 = "body 横向侧移 + 同向
            # yaw"(TURN 实测 body=(0,0.4)+wz=0.4 全程有效;旧实现后退+弱
            # 侧移 0.13+弱转 0.30 无效,R8j 卡死实证)。侧移命令沿 body 横
            # 轴发出(投影回 body 恒为纯侧移),wz 用 2.0×error 强增益且带
            # 下限。x 东漂护栏:超 drift_limit 先切 settle 修正,防贴
            # 东墙(x=-1.74)。同时全程 x 实时伺服对中(掉头打滑时护栏太
            # 迟——R20 实证 x 滑出 -0.5 前 z 已骤降,恢复/settle 都救不回
            # 物理卡死;实时对中把打滑漂移约束在安全区内)。
            if turn_done:
                self.segment_turn_stage='settle'
                rospy.loginfo('Segment turn heading done (err=%.3f rad); '
                              'settling position.', error)
            elif x_err < -self.segment_turn_drift_limit:
                self.segment_turn_stage='settle'
                rospy.logwarn('Segment turn drifted x=%.3f east; settling '
                              'position before continuing turn.',
                              self.truth_pose[0])
            else:
                yaw_rate=self._segment_turn_strong_rate(turn_error)
                # R37 慢启动:turn 开始头 2s 转体速度渐进,对齐 TURN 修复 J
                # 的慢启动(大步态起步即滑落)。起点 0.24(策略 yaw 死区
                # 上缘,TURN 修复实证 0.10 起步转不动),2s 渐进到满速。
                if self.segment_turn_turn_started_at is not None:
                    ramp_since=(now-self.segment_turn_turn_started_at)
                    if 0.0 <= ramp_since < 2.0:
                        cap=(0.24+(self.segment_turn_yaw_rate-0.24)*
                             ramp_since/2.0)
                        yaw_rate=max(-cap, min(cap, yaw_rate))
                lat=(self.segment_turn_lateral_speed
                     if yaw_rate >= 0.0 else -self.segment_turn_lateral_speed)
                # world 投影 = R(yaw)·(0, lat),投影回 body 恒为纯侧移
                vx=-lat*math.sin(self.truth_pose[3])
                vy=lat*math.cos(self.truth_pose[3])
                # x 保持:掉头全程对中 flight_b 中心线,防东滑进 flight_b
                # 起点阶梯(R20 实证)
                x_hold=max(-self.landing_recenter_speed,
                           min(self.landing_recenter_speed,
                               self.landing_position_gain*x_err))
                vx+=x_hold
                # y 保持:安全区间 [y_low, y_high] 内零命令(区间外才拉
                # 回边界)。R27 实证 y 伺服在 2.24 打滑推不动,固定目标
                # 伺服会把掉头卡死在 align;掉头只需 y 在平坦带内。
                if self.truth_pose[1] < self.segment_turn_y_low:
                    vy+=max(-self.segment_turn_side_speed,
                            min(self.segment_turn_side_speed,
                                self.landing_position_gain*
                                (self.segment_turn_y_low-
                                 self.truth_pose[1])))
                elif self.truth_pose[1] > self.segment_turn_y_high:
                    vy+=max(-self.segment_turn_side_speed,
                            min(self.segment_turn_side_speed,
                                self.landing_position_gain*
                                (self.segment_turn_y_high-
                                 self.truth_pose[1])))
                self._publish_truth_world_command(vx, vy, yaw_rate)
                self._record_trace()
                return
        # 阶段 3:位置微调 + 保持航向(body 侧移修正 x/y,PRE_ALIGN 同源)
        if not turn_done:
            self.segment_turn_stage='turn'
            return
        vx=max(-self.landing_recenter_speed,
               min(self.landing_recenter_speed,
                   self.landing_position_gain*x_err))
        if abs(vx) < self.landing_minimum_x_speed:
            vx=math.copysign(self.landing_minimum_x_speed, vx)
        vy=0.0
        if self.truth_pose[1] < self.segment_turn_y_low:
            vy=max(-self.segment_turn_side_speed,
                   min(self.segment_turn_side_speed,
                       self.landing_position_gain*
                       (self.segment_turn_y_low-self.truth_pose[1])))
        elif self.truth_pose[1] > self.segment_turn_y_high:
            vy=max(-self.segment_turn_side_speed,
                   min(self.segment_turn_side_speed,
                       self.landing_position_gain*
                       (self.segment_turn_y_high-self.truth_pose[1])))
        wz=max(-self.entry_alignment_yaw_rate,
               min(self.entry_alignment_yaw_rate, .90*error))
        if abs(wz) < self.landing_minimum_yaw_rate:
            wz=math.copysign(self.landing_minimum_yaw_rate, wz)
        self._publish_truth_world_command(vx, vy, wz)
        self._record_trace()

    # ------------------------------------------------------------------ #
    # 段完成
    # ------------------------------------------------------------------ #
    def _segment_complete(self):
        drop=self._drop()
        last_segment=(self.segment+1 >= self.total_segments)
        if not last_segment:
            # F3→F2 完成:写段 1 终态 JSON,段间转体后继续段 2
            self._write_segment_json(
                self.segment_one_log_basename, 'SECOND_FLOOR_DESCENT_REACHED')
            self._record_trace(force=True)
            self.segment_one_elapsed_sec=round(self._segment_elapsed(), 3)
            self._publish_milestone('F2_LANDING_REACHED')
            rospy.loginfo(
                'Segment 1 (F3->F2) complete: drop=%.2f m, elapsed %.1f s; '
                'turning on F2 landing for segment 2.',
                drop, self.segment_one_elapsed_sec)
            self.segment=1
            self._begin_segment()
            self.state.publish(String(data=self.phase))
            return
        if self.end_floor_number == 2:
            # 隔离 harness 单段(F3→F2):到达 2 层即终态
            self._write_segment_json(
                self.segment_one_log_basename, 'SECOND_FLOOR_DESCENT_REACHED')
            self._record_trace(force=True)
            rospy.loginfo(
                'Descent to floor 2 complete after %.1f s (drop=%.2f m).',
                self._elapsed(), drop)
            self._terminal('SECOND_FLOOR_DESCENT_REACHED',
                           'second_floor_descent_reached')
            return
        # F2→F1 只代表到达一楼 landing，不再冒充“回到出生点”。保留兼容
        # 文件，但以明确中间态记录，然后换回 plane policy 继续返航。
        self.first_floor_landing_ros_time=rospy.Time.now().to_sec()
        self._publish_milestone('F1_LANDING_REACHED')
        self._write_segment_json(self.segment_two_log_basename,
                                 'FIRST_FLOOR_LANDING_REACHED')
        self._record_trace(force=True)
        rospy.loginfo('First-floor landing reached after %.1f wall s '
                      '(drop=%.2f m); continuing to original start.',
                      self._elapsed(), drop)
        try:
            with open(os.path.join(self.out, 'first_floor_returned.json'),
                      'w') as stream:
                json.dump({
                    'phase': 'FIRST_FLOOR_LANDING_REACHED',
                    'wall_time': round(time.time(), 3),
                    'ros_time': round(self.first_floor_landing_ros_time, 3),
                    'trigger_ros_time': self.trigger_ros_time,
                    'descent_ros_seconds': (
                        round(self.first_floor_landing_ros_time-
                              self.trigger_ros_time, 3)
                        if self.trigger_ros_time is not None else None),
                    'elapsed_wall_sec': round(self._elapsed(), 3),
                    'descent_total_drop': round(drop, 3),
                    'segment_one_phase': 'SECOND_FLOOR_DESCENT_REACHED',
                    'segment_one_elapsed_sec': self.segment_one_elapsed_sec,
                }, stream, indent=2)
        except OSError as error:
            rospy.logerr('Cannot write first_floor_returned.json: %s', error)
        self.first_floor_state_pub.publish(
            String(data='FIRST_FLOOR_LANDING_REACHED'))
        self._begin_home_return()

    # ------------------------------------------------------------------ #
    # 保存
    # ------------------------------------------------------------------ #
    def _write_segment_json(self, basename, phase):
        try:
            with open(os.path.join(self.out, 'logs', basename), 'w') as stream:
                json.dump({'phase': phase,
                           'policy': self.policy,
                           'wall_time': round(time.time(), 3),
                           'ros_time': round(rospy.Time.now().to_sec(), 3),
                           'trace': self.trace},
                          stream)
        except OSError as error:
            rospy.logerr('Cannot write %s: %s', basename, error)

    def save(self):
        # Clean one-shot controller-boundary parameters on external launch
        # shutdown as well, so a failed run cannot poison the next run.
        self._clear_fast_fixed_stand()
        self._clear_stand_target()
        self._clear_fast_takeover()
        try:
            with open(os.path.join(self.out, 'logs',
                                   self.transition_log_basename), 'w') as stream:
                json.dump({'phase': self.phase,
                           'policy': self.policy,
                           'wall_time': round(time.time(), 3),
                           'trigger_ros_time': self.trigger_ros_time,
                           'milestones': self.milestones,
                           'stall_recovery_count':
                               self.flight_stall_recoveries,
                           'edge_drift_landing_capture_count':
                               self.edge_drift_landing_capture_count,
                           'fall_recovery_count':
                               self.fall_recovery_count,
                           'post_trigger_model_state_reset_count':
                               self.post_trigger_model_state_reset_count,
                           'stand_target_evidence':
                               self.stand_target_evidence,
                           'entry_guide': {
                               'two_stage_guide': self.truth_two_stage_guide,
                               'stage_switches': self.truth_entry_stage_switches,
                               'stage': self.truth_entry_stage,
                           },
                           'trace': self.trace},
                          stream)
        except OSError as error:
            rospy.logerr('Cannot write %s: %s',
                         self.transition_log_basename, error)


def main():
    rospy.init_node('stair_descent_manager')
    node=StairDescent()
    rospy.spin()


if __name__ == '__main__':
    main()
