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
import json, math, os, time, shutil
import xml.etree.ElementTree as ET
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Joy
from std_msgs.msg import String, Bool
from gazebo_msgs.msg import ModelStates, LinkStates
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
        self.trace=[]; self.last_log=0.
        self.policy_loaded=False; self.locomotion_ready=False
        self.odom_seen=False; self.policy_requested=False
        self.segment_one_elapsed_sec=None
        self._pause_active=False

        # ---- 下梯速度与超时 ----
        self.descent_speed=float(rospy.get_param('~descent_speed_mps', .35))
        self.descent_maximum_speed=float(rospy.get_param(
            '~descent_maximum_speed_mps', .40))
        self.descent_segment_timeout=float(rospy.get_param(
            '~descent_segment_timeout_sec', 180.0))
        self.trigger_timeout=float(rospy.get_param(
            '~truth_descent_trigger_timeout_sec', 300.0))
        # 后向下梯:目标朝向 = 各 flight 的上行方向(背对下行方向),
        # 世界系命令不变,投影自动产生 body 负向速度(与 flight-b backoff 同源)。
        self.backward_mode=bool(rospy.get_param(
            '~truth_descent_backward_mode', False))
        self.pre_descent_stand_seconds=float(rospy.get_param(
            '~pre_descent_stand_seconds', 8.0))
        self.pre_descent_stand_started=None
        self.pre_descent_stand_done=False

        # ---- 入口引导(F3 走廊口 → F3 landing)----
        self.truth_entry_guide=bool(rospy.get_param('~truth_entry_guide', True))
        self.truth_two_stage_guide=bool(rospy.get_param(
            '~truth_two_stage_guide', True))
        self.offline_stair_model=rospy.get_param('~offline_stair_model_sdf', '')
        self.offline_layout=rospy.get_param('~offline_truth_layout_metadata', '')
        # R41b:layout 解析保存(原仅复制)。入口引导用房间 bounds/门洞
        # 判定"起点在房间内"并生成走廊重入点。
        self.truth_offline_layout=None
        if self.offline_layout and os.path.isfile(self.offline_layout):
            try:
                with open(self.offline_layout) as fh:
                    self.truth_offline_layout=json.load(fh)
            except (OSError, ValueError):
                self.truth_offline_layout=None
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

        # ---- 高速下梯速度整形(迁移自 scanplanner 方案,2026-09-01) ----
        # zip 方案(stair_descent_controller.launch)以 0.80 m/s 巡航下梯,
        # 但按本地垂直进度整形:首段接触 0.55 m/s(hold 到 drop 0.18m),
        # 线性爬升到 0.80(drop 0.45m 后),末端 drop≥0.95m 收 0.60。
        # 我们的 0.48 恒定速度无整形,0.50 全流程即出现随机噪声(2026-08-27
        # 定案);整形的关键不是峰值速度,而是起步/收尾/倾斜时提前减速。
        self.flight_initial_speed=max(.30, float(rospy.get_param(
            '~truth_descent_flight_initial_speed_mps', .55)))
        self.flight_initial_hold_drop=max(0.0, float(rospy.get_param(
            '~truth_descent_flight_initial_hold_drop_m', .18)))
        self.flight_initial_ramp_end_drop=max(
            self.flight_initial_hold_drop, float(rospy.get_param(
                '~truth_descent_flight_initial_ramp_end_drop_m', .45)))
        self.flight_end_slowdown_drop=max(0.0, float(rospy.get_param(
            '~truth_descent_flight_end_slowdown_drop_m', .95)))
        self.flight_end_speed=max(.30, float(rospy.get_param(
            '~truth_descent_flight_end_speed_mps', .60)))
        # 倾角预判降速:正常下梯姿态 roll/pitch 约 0.45 rad;中间档
        # 0.48 起保持 0.55 m/s 接触速度,严重档 0.60 立即跌到 0.30。
        self.flight_tilt_slowdown=max(0.0, float(rospy.get_param(
            '~truth_descent_flight_tilt_slowdown_rad', .48)))
        self.flight_tilt_speed=max(.30, float(rospy.get_param(
            '~truth_descent_flight_tilt_speed_mps', .55)))
        self.flight_tilt_hold=max(0.0, float(rospy.get_param(
            '~truth_descent_flight_tilt_hold_sec', .60)))
        self.flight_severe_tilt_slowdown=max(0.0, float(rospy.get_param(
            '~truth_descent_flight_severe_tilt_rad', .60)))
        self.flight_severe_tilt_speed=max(.30, float(rospy.get_param(
            '~truth_descent_flight_severe_tilt_speed_mps', .30)))
        self.flight_severe_tilt_hold=max(0.0, float(rospy.get_param(
            '~truth_descent_flight_severe_tilt_hold_sec', .80)))
        self.flight_tilt_slow_until=None
        self.flight_severe_tilt_until=None
        # 误差修正降速:中心误差 ≥0.22m 或航向误差 ≥0.22rad 时以更高
        # yaw 率修正、同步压低轴速,避免漂移在硬守卫(0.45)前失控。
        self.flight_center_correction_error=max(0.0, float(rospy.get_param(
            '~truth_descent_flight_center_correction_error_m', .22)))
        self.flight_center_correction_speed=max(.30, float(rospy.get_param(
            '~truth_descent_flight_center_correction_speed_mps', .50)))
        self.flight_center_correction_yaw_rate=max(0.0, float(rospy.get_param(
            '~truth_descent_flight_center_correction_yaw_rate_rps', .20)))
        self.flight_heading_correction_error=max(0.0, float(rospy.get_param(
            '~truth_descent_flight_heading_correction_error_rad', .22)))
        self.flight_heading_correction_speed=max(.30, float(rospy.get_param(
            '~truth_descent_flight_heading_correction_speed_mps', .40)))
        self.flight_heading_correction_yaw_rate=max(
            0.0, float(rospy.get_param(
                '~truth_descent_flight_heading_correction_yaw_rate_rps',
                .30)))

        # ---- 中间平台转体 ----
        # full8 2026-08-17:完成门 4.70/4.80 让转体点距 0.27m 平台台阶仅
        # 3cm,转体打滑滑下台阶 → y 安全带冻结 → TURN_TIMEOUT。默认对齐
        # 注释意图与隔离 harness 的 5.05(平台中心,28cm 滑落缓冲)。
        self.turn_start_y=float(rospy.get_param(
            '~truth_descent_turn_start_y', 5.05))
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
            '~truth_descent_landing_minimum_x_speed_mps', .10))
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
            '~truth_descent_landing_slip_max_attempts', 6)))
        # R47(2026-08-28):SEGMENT_TURN align 位置停滞检测。命令已发出但
        # 位置几乎不动 = 步态打滑/楔入 tread 过渡带台阶。R46 batch18 实证:
        # 落点 y=2.165-2.181 仅低于门 2.20 且 z 已落 2.83 时,z_drop 检测
        # 失效(landing_z0 在跌落完成后捕获,无骤降),比例推 y 落入步态
        # 死区冻结 45s → 3/3 SEGMENT_TURN_TIMEOUT。死区下限
        # (_segment_turn_side_vy)修掉死区冻结后,残留的物理楔入由本检测
        # 兜底 → 落步+回退+重植(与 slip recovery 同款恢复链)。
        self.segment_turn_align_stall_seconds=float(rospy.get_param(
            '~truth_descent_segment_turn_align_stall_sec', 3.0))
        self.segment_turn_align_stall_tolerance=float(rospy.get_param(
            '~truth_descent_segment_turn_align_stall_tol_m', .02))
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
        # R39: 掉头期间南缘护栏。'turn' 阶段只有 x 伺服没有 y 伺服,转体
        # 步态侧滑会把机器人南漂出平台南缘(y<2.05 已是 flight_b 阶梯
        # tread,z 骤降 0.45m;R20/R21 实证滑落侧翻 roll≈2.65-3.1 rad
        # 物理不可恢复)。y 低于护栏即暂停转体,world +y 推回 guard_recenter
        # (掉头安全带北半平坦区,该区 x≈-3.7 处实测 z 恒平 3.04,无台阶)
        # 再继续(R31 landing TURN 同款护栏;北侧不设护栏 —— R27 实证
        # 向北推 y>2.05 的 flight_a step 会打滑冻结)。
        self.segment_turn_y_guard=float(rospy.get_param(
            '~truth_descent_segment_turn_y_guard_m', 2.08))
        # R45(2026-08-26):转体阶段护栏线。batch8 RUN3 实证转体步态侧滑
        # 把 y 从 2.24 一路滑到 2.09(z 3.10→2.87 深陷),旧护栏 2.08
        # 触发太晚(深陷后推不回)。转体期间 y 跌破 2.15(z 尚高、推 y
        # 有效)就暂停转体推回,不等深陷。
        self.segment_turn_turn_y_guard=float(rospy.get_param(
            '~truth_descent_segment_turn_turn_y_guard_m', 2.15))
        # R56(2026-08-29 批次9):北缘护栏 —— 批次8 run2/run4+批次9
        # run1/run4 实证掉头转体步态失效时 y 狂漂北(2.9→8.5 只转 28°),
        # yaw stall 6s 才触发恢复,recovery 重植后 align 冻结(y=3.85/
        # 2.499/1.99)4 连发 SEGMENT_TURN_TIMEOUT。R45 只防南滑(2.15),
        # 北侧无上限 —— y>2.45 暂停转体,world -y 推回掉头带(2.30)再
        # 继续。R27 的"北侧不设护栏"是防向北推(y 低推北),此处方向
        # 相反(推南),run1 实证 y 漂到 8.5 时 z 恒 2.91 仍平地,推回有效。
        self.segment_turn_turn_y_guard_north=float(rospy.get_param(
            '~truth_descent_segment_turn_turn_y_guard_north_m', 2.45))
        # R58(2026-08-29 批次11 run2):护栏推回无效检测 —— 转体步态失效
        # 时侧移也失力(R45 南护栏推回 4s y 2.15→1.996 反滑),z 跌 0.12
        # 才触发 slip recovery 时 y 已滑出 2.05 安全带楔入 flight_b 阶梯,
        # 6 次 recovery 救不回。推回 ≥2.5s 且净回升 <0.05m 判定推回无效
        # → 提前 recovery(y 尚在安全带内时重植,位置好救)。
        self.segment_turn_push_ineffective_sec=float(rospy.get_param(
            '~truth_descent_segment_turn_push_ineffective_sec', 2.5))
        self.segment_turn_push_min_recover_m=float(rospy.get_param(
            '~truth_descent_segment_turn_push_min_recover_m', 0.05))
        # R43(2026-08-26):2.25→2.20。batch8 RUN1 实证 settle/align 推 y
        # 把 y 从 2.047 推到 2.22 后推不动(后腿顶 flight_a 末级 tread
        # 过渡带上沿),2.25 门不可达必超时;2.22 是推 y 物理上限,留
        # 0.02 余量取 2.20。batch6 RUN2 成功轮转体起点 y=2.19-2.27 稳定
        # 佐证 2.20 区间可站可转。推 x 的安全 y(2.25)不可达 → align 不再
        # 推 x(见 align 门 vx=0 逻辑),x 修正交给 turn/settle(转体后 z
        # 升高,b6R2 settle 在 z=3.10 推 x 0.36m 成功)。
        self.segment_turn_y_guard_recenter=float(rospy.get_param(
            '~truth_descent_segment_turn_y_guard_recenter_m', 2.20))
        # y 安全区间(landing 平坦带):y_keep±0.35。R27 实证 y 伺服把 y 从
        # 2.17 推到 2.24 后步态打滑推不动(冻结 37.5s,恢复 4 次无效,
        # SEGMENT_TURN_TIMEOUT)——y 只需在平坦带内即可掉头,不必收敛到
        # 精确 2.4;区间外才拉回,区间内零命令(y_keep 伺服打滑会卡死)。
        # full12 实证:掉头完成时 y 南漂到 2.072,±0.3 带(y_low 2.1)判
        # 失败 → settle 死循环超时;放宽下界到 2.05(flight_b tread 边缘
        # 再低 2cm,z 掉 0.45m,slip 检测 z_drop>0.12 兜底)。
        self.segment_turn_y_band=float(rospy.get_param(
            '~truth_descent_segment_turn_y_band_m', 0.35))
        self.segment_turn_y_low=self.segment_turn_y_keep-\
            self.segment_turn_y_band
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
        # 段 2 起步东移对齐目标(F2 landing 平台内 y=2.0,x 对齐
        # flight_b_floor_0 中心线 -2.485;平台 y∈[1.05,2.05],北缘
        # y>2.05 已是楼梯)。
        self.east_align_y_target=float(rospy.get_param(
            '~truth_descent_east_align_y_m', 1.85))
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
        # R60(2026-08-29):批次12 run3/run4 实证 EAST_ALIGN 渐进北漂型
        # TIP_OVER ×2 连发 —— roll 0.54/0.56 触发北退后 0.77/0.83s 翻倒
        # (≈R52 |roll|>0.5 持续 0.8s 判定窗口)。北退推西 1.2s 尚未生效
        # roll 已判死;对比 run2 roll=0.51 窗口内回落救回。双杠杆:
        #   a) north_drift_roll 0.5→0.40 —— 更早拦截,roll 未临界时西退
        #      运动有生效时间(正常侧移 roll 恒 <0.3,0.40 不误触发)
        #   b) backoff 期间 roll_bad 判定窗口 0.8→2.5s —— 给北退
        #      1.2s+extend 余量;|roll|>0.9 瞬时兜底在 backoff 中保持
        self.east_align_north_drift_roll=float(rospy.get_param(
            '~truth_descent_east_align_north_drift_roll', 0.40))
        self.east_align_roll_bad_sec=float(rospy.get_param(
            '~truth_descent_east_align_roll_bad_sec', 0.8))
        self.east_align_roll_bad_backoff_sec=float(rospy.get_param(
            '~truth_descent_east_align_roll_bad_backoff_sec', 2.5))
        # R50(2026-08-28):批次3 实证 R49 shelf_z=2.96 致命回归 —— 平台
        # z∈[2.96,3.09],east 起步即 z≤shelf_z → vy=0 纯东移 + y-priority
        # 失效,北漂失控侧翻(roll 1.24/2.13,0/5)。回退:两处阈值统一用
        # tread_z=2.75 —— z∈(2.75,3.10] 全程保留 y 伺服+y-priority
        # (批次2 3/5 成功路径原行为);仅 z≤2.75 真正入 tread 才纯东移
        # (治批次2 run1 对角死区冻结)。shelf_z 参数删除。

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
            '~truth_descent_flight_max_stall_recoveries', 30))
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
        # R67(2026-08-30 批次24):SEGMENT_TURN 绝对 z 陷落交棒计时 ——
        # 批次24 实证相对 z_drop(landing_z0 差)在三种场景失效(run4
        # 降落即低起点 2.684 / run5 re-planted 竞态写低 2.737 / run2
        # 掉一层 0.388),z≤tread_z(2.75)失力临界与起点无关,直接计时
        self.segment_turn_low_z_at=None
        self.segment_turn_low_z_instant=float(rospy.get_param(
            '~truth_descent_segment_turn_low_z_instant', 2.50))
        self.segment_turn_low_z_hold=float(rospy.get_param(
            '~truth_descent_segment_turn_low_z_hold', 3.0))
        # R68:SEGMENT_TURN 翻倒检测(R50/R52 同款)与交棒前 yaw 对齐
        self.segment_turn_roll_bad_at=None
        self.segment_turn_prep_yaw_until=None
        # R69(2026-08-31 批次28 run2):PRE_ALIGN 转体 watchdog 状态
        # (run2 实证纯转体 (0,0,wz) 冻结 15.6min 无超时 → NO_TRACE)
        self.pre_align_turn_started_at=None
        self.pre_align_yaw_anchor=None
        self.pre_align_recover_until=None
        self.pre_align_recoveries=0
        self.landing_slip_active=False
        self.landing_slip_until=None
        self.landing_slip_backoff_until=None
        self.landing_slip_recoveries=0
        # R57(2026-08-29 批次10 run5):backoff 方向/vy 自适应(见 2261 段
        # —— 固定 east 在掉头东滑场景把 x 越推越东,backoff 需同时修 y)
        self.landing_slip_backoff_dir=1.0
        self.landing_slip_backoff_vy=0.0
        # R58:护栏推回无效检测计时(见 turn 阶段南/北护栏分支)
        self.segment_turn_push_started_at=None
        self.segment_turn_push_start_y=None
        # R61(2026-08-29 批次14 run5):backoff 失效升级 + y 带外持续修正
        # (见 backoff 结束块 —— 推不动时距离升级,y 带外时延长修正)
        self.segment_turn_backoff_origin=None
        self.segment_turn_backoff_escalations=0
        self.segment_turn_backoff_ext=0
        # R62(2026-08-29 批次15 run5):backoff 升级距离累乘变量(R61 bug:
        # distance*1.5 每次用参数 0.55 → 两次升级都 0.83m,应 0.83/1.24)
        self.segment_turn_backoff_dist=None
        self.landing_yaw_stall_accum=0.0
        self.landing_last_yaw=None
        self.landing_last_yaw_at=None
        self.segment_turn_stall_pos=None
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
        rospy.Timer(rospy.Duration(.02), self.tick)
        rospy.on_shutdown(self._on_shutdown_save)

    # ------------------------------------------------------------------ #
    # 回调
    # ------------------------------------------------------------------ #
    def on_mission_trigger(self, message):
        token=str(message.data).strip()
        if self.phase == 'WAIT_F3' and token == self.mission_trigger_token:
            self.started=time.monotonic()
            self.trigger_wall_time=time.time()
            self.segment_started=self.started
            self._begin_segment()
            self.state.publish(String(data=self.phase))

    def on_policy_status(self, message):
        if ('policy_reloaded:' in message.data and
                os.path.basename(self.policy) in message.data):
            self.policy_loaded=True
            self.state.publish(String(data='STAIR_DESCENT_POLICY_READY'))
        elif 'policy_reload_failed:' in message.data and \
                self.phase == 'STAIR_DESCENT_POLICY_LOADING':
            self._terminal('STAIR_DESCENT_POLICY_FAILED',
                           'stair_descent_policy_failed', error=True)

    def on_locomotion_ready(self, message):
        self.locomotion_ready=bool(message.data)

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
            self._drop(), self.segment,
            math.degrees(self.truth_attitude[1]) if self.truth_attitude
            else float('nan'))
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
        if self.truth_pose is None:
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
              'wall_time': round(time.time(), 3),
              'phase': self.phase,
              'x': pose[0], 'y': pose[1], 'z': pose[2], 'yaw': pose[3],
              'segment': self.segment,
              'descent_drop': round(self._drop(), 3),
              'stall_recoveries': self.flight_stall_recoveries,
              'side_slip_attempts': self.side_slip_attempts,
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
        if self.phase == 'STAIR_DESCENT_ENTRY_GUIDE':
            item.update({'entry_stage': self.truth_entry_stage,
                         'target_distance': self.truth_entry_last_distance})
        self.trace.append(item)
        self.last_log=now

    def _terminal(self, phase, shutdown_reason, error=False):
        self.cmd.publish(Twist())
        self.hold_rl()
        self.phase=phase
        self.state.publish(String(data=self.phase))
        if error:
            rospy.logerr('Stair descent terminal phase: %s', phase)
        else:
            rospy.loginfo('Stair descent terminal phase: %s', phase)
        self._record_trace(force=True)
        self.save()
        rospy.signal_shutdown(shutdown_reason)

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
            '~truth_descent_flight_max_stall_recoveries', 30))
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
            # R41b:探索异常收尾(TIME_LIMIT/GOAL_FAILURE)可能把机器人留
            # 在房间内,直线引导会穿墙/冻结;房间内起点先重入走廊。
            self.truth_reenter_points=self._truth_entry_reenter_route()
            self.truth_entry_roll_bad_at=None
            self.phase='STAIR_DESCENT_ENTRY_GUIDE'
        elif self.segment == 0:
            # 跳过入口引导时也要先 PRE_ALIGN:在 landing 上把 x 横移到
            # flight 中心线并原地转体到下行朝向,再加载 stair 策略。
            # (2026-08-15 冒烟实证:跳过此步直接 POLICY_LOADING,
            #  0.765m 横向偏差 → heading-bias 饱和 0.12/0.30 → 带 ~29°
            #  斜向下梯,第一级真实 riser 处栽倒。)
            # R69:重置转体 watchdog 状态(恢复路径可能重进 PRE_ALIGN)
            self.pre_align_turn_started_at=None
            self.pre_align_yaw_anchor=None
            self.pre_align_recover_until=None
            self.pre_align_recoveries=0
            self.phase='STAIR_DESCENT_PRE_ALIGN'
        else:
            self.phase='STAIR_DESCENT_SEGMENT_TURN'
            self.segment_turn_started_at=time.monotonic()
            self.segment_turn_stage='align'
            self.segment_turn_direction=None
            self.segment_turn_turn_started_at=None
            self.segment_turn_stall_pos=None
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
            # R67:段起点重置绝对 z 计时;R68:重置翻倒与 yaw 对齐状态
            self.segment_turn_low_z_at=None
            self.segment_turn_roll_bad_at=None
            self.segment_turn_prep_yaw_until=None

    def _enter_policy_loading(self):
        """请求 stair 策略并等待 policy_reloaded。"""
        self.phase='STAIR_DESCENT_POLICY_LOADING'
        if not self.policy_requested:
            self.policy_requested=True
            self.pub.publish(String(data=self.policy))
            rospy.loginfo('Descent requesting stair policy: %s', self.policy)
        self.cmd.publish(Twist())
        self.hold_rl()

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
        # R49:east 段位置停滞检测(死区冻结兜底,同 south fixX 模式)
        self.east_align_east_stall_at=None
        self.east_align_east_stall_pos=None
        self.east_align_east_retries=0
        self.east_align_east_backoff_until=None
        # R54:北退南修延长计数(见 _east_align_tick backoff 分支)
        self.east_align_east_backoff_ext=0
        # R65:z 陷落深回退状态(见 _east_align_tick east 段,retry 耗尽兜底)
        self.east_align_deep_backoff_until=None
        self.east_align_deep_backoff_count=0
        # R70(2026-08-31 批次32 run2):retry/深回退耗尽后的静止重置轮计数
        # 与强制深回退标志(见 _east_align_tick east 段耗尽分支)
        self.east_align_retry_rounds=0
        self.east_align_force_deep=False
        # R50:翻倒检测状态(批次3 run1/2/5 平台侧翻 roll 1.24-2.13 rad,
        # 翻后任何命令无效,30s 死等超时纯浪费)
        self.east_align_roll_bad_at=None
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
        # R67:超时 30→45s —— 批次24 run3 实证一次北退(1.2s)+extend
        # 4/4(2.4s)+resume 停滞(10s)+第二次北退+深回退(10s)≈35s 超
        # 30s 预算;成功轮 13-23s 完成,45s 给北退/深回退链留余量。
        # fixU(z 门交棒)/TIP_OVER 兜底不变,卡死轮仅多等 15s。
        # R70:45→55s —— 耗尽重置轮(静止 10s + backoff + resume 东移)需要
        # ~45-50s 预算(批次32 run2:70.5s 起 93s 冻结,重置后 ~110s 可完成,
        # 45s 超时差 6s);成功轮 6.9-7.6s 完成,55s 仅影响卡死轮多等 10s。
        if now-self.east_align_started >= 55.0:
            self._terminal('STAIR_DESCENT_EAST_ALIGN_TIMEOUT',
                           'stair_descent_east_align_timeout', error=True)
            return
        # R50:平台翻倒检测 —— |roll|>0.6 rad(34°)持续 1s 判定不可恢复
        # 侧翻(正常侧移 roll 恒 <0.3,批次3 翻倒达 1.24-2.13)。命令全零
        # + 干净报错,不再 30s 死等。同入口段 R44 姿态护栏模式。
        # R52:批次5 run3 实证边缘滑落侧翻中 roll 震荡(-0.62/-0.64/-0.92
        # 被 -0.46/-0.51 间歇打断),旧"连续 1s"条件漏检 → 改为双重阈值:
        #   a) |roll|>0.9 rad(52°)瞬时判翻倒 —— 正常侧移恒 <0.3,此值
        #      只在真实倾倒中出现(批次5 run3 t=142.0 roll=-1.21)
        #   b) |roll|>0.5 持续 0.8s —— 捕捉震荡型滑落(0.8s 窗口内
        #      间歇 <0.2s 不重置,run3 141.9-142.4 连续 0.5s 仍由 a 兜底)
        if abs(self.truth_attitude[0]) > 0.9:
            self._terminal('STAIR_DESCENT_EAST_ALIGN_TIP_OVER',
                           'stair_descent_east_align_tip_over',
                           error=True)
            return
        if abs(self.truth_attitude[0]) > 0.5:
            if self.east_align_roll_bad_at is None:
                self.east_align_roll_bad_at=now
            # R60:北退 backoff 期间放宽判定窗口(0.8→2.5s)——
            # 批次12 run3/4 实证 roll 0.54/0.56 触发北退后 0.8s 内
            # roll 尚未回落即判死(西退 1.2s 运动未生效),放宽给足
            # backoff 时间;backoff 外保持 0.8s。|roll|>0.9 瞬时兜底
            # 不受影响(真翻倒仍在 backoff 中拦截)。
            elif now-self.east_align_roll_bad_at >= (
                    self.east_align_roll_bad_backoff_sec
                    if self.east_align_east_backoff_until is not None
                    else self.east_align_roll_bad_sec):
                self._terminal('STAIR_DESCENT_EAST_ALIGN_TIP_OVER',
                               'stair_descent_east_align_tip_over',
                               error=True)
                return
        else:
            self.east_align_roll_bad_at=None
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
            # R49:south→east 门收紧 0.05/0.10 → 0.03/0.08 —— 批次2 run1
            # 实证 y=1.94(|y_err|=0.09<0.10)放行后北漂进北缘(z 早跌)冻
            # 结;y≤1.88 在东移 1.1m(北漂耦合 ~0.11)后仍 ≤1.99。
            if (self.truth_pose[1] <= self.east_align_y_target+0.03 and
                    abs(y_err) < 0.08):
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
            # R48(2026-08-28):批次1 run5 实证 z 跌破 tread_z(2.75)进台阶
            # 后 y-priority 仍 vx=0 纯南推(0.40)+yaw 强伺服(wz=-1.0)→ 斜面
            # 上冻结 20s(EAST_ALIGN_TIMEOUT)。fixV3 本意是"z 触发但 x 未
            # 到位时继续东移" —— y-priority 只在平台平地(z>tread_z)生效;
            # 进 tread 后东移单调向中心拉近,fixU 交棒阈值 x_err≤0.35 兜底。
            # R50:批次3 实证 R49 的 shelf_z=2.96 直接打死 y 伺服(平台
            # z≈2.96 起步即触发) → 北漂侧翻。恢复 R48:tread_z=2.75 以上
            # 平台平地全程保留 y-priority + y 伺服;仅 z≤2.75 真入 tread
            # 后 vy=0 纯东移(对角死区冻结的批2 run1 由收紧门 y≤1.88 防)。
            # R52:预警 1.95→1.90(批次5 run3 实证 y 从 1.89 到 2.05 仅
            # 0.4s,1.95 触发时 z 已跌近 tread_z,南修窗口 <0.2s)。
            if self.truth_pose[1] > 1.90 and \
                    self.truth_pose[2] > self.east_align_tread_z:
                vx=0.0
                vy=max(-self.segment_turn_lateral_speed,
                       min(self.segment_turn_lateral_speed,
                           self.landing_position_gain*y_err))
                if abs(y_err) > 0.03 and \
                        abs(vy) < self.segment_turn_lateral_speed:
                    vy=math.copysign(self.segment_turn_lateral_speed, vy)
            # R53:北退与 z 门解耦 —— 批次6 5 轮实证 EAST_ALIGN 平台
            # z 恒 ≥2.85 从未跌破 tread_z(2.75),R52 的"z≤tread_z 且
            # y>1.95"从未触发,形同虚设。真实危险信号:east 段东移中
            # y 北漂且 roll>0.5 = 南修已失效的边缘打滑(run2 实证 roll
            # 0.71→0.95 翻倒;run3 同场景 roll 0.68-0.79 碰运气恢复)。
            # roll<0.5 的北漂(y 至 2.16)南修仍有效(run4 实证拉回),
            # 不北退。south 段 y>2.0 是正常南修,必须排除(stage 门)。
            # R60:阈值 0.5→0.40 —— 批次12 run3/4 实证 roll 0.54/0.56
            # 才触发时北退已救不回(西退运动未生效即判死),提前到 0.40
            # 在步态失稳早期拦截(正常侧移 roll 恒 <0.3,0.40 不误触发)。
            # R54:backoff 期间不再重复触发 —— R53 原判定不看
            # backoff_until,backoff 中 roll 仍 >0.5 时每个 tick 都 +1,
            # 1/2、2/2 配额在 ~0.2s 内烧光(run4 实证相邻两行),导致
            # 2 次后无兜底。
            if (self.east_align_stage == 'east' and
                    self.east_align_east_backoff_until is None and
                    self.truth_pose[1] > 1.95 and
                    (abs(self.truth_attitude[0]) >
                     self.east_align_north_drift_roll or
                     self.truth_pose[2] <= self.east_align_tread_z)):
                # 复用 R49 西退机制(1.2s≈0.4m 回平台平地),上限 3 次
                # 与停滞西退共享;耗尽后靠 fixU 兜底+roll 检测。
                if self.east_align_east_retries < 3:
                    self.east_align_east_retries+=1
                    self.east_align_east_backoff_until=now+1.2
                    self.east_align_east_backoff_ext=0
                    self.east_align_east_stall_at=None
                    rospy.logwarn(
                        'East-align north drift z=%.2f y=%.2f roll=%.2f; '
                        'backing off west, retry %d/3.',
                        self.truth_pose[2], self.truth_pose[1],
                        self.truth_attitude[0], self.east_align_east_retries)
            if self.truth_pose[2] <= self.east_align_tread_z:
                vy=0.0
            # R49:east 段位置停滞检测 —— 3s 窗口位移<0.02m 判定死区冻结
            # (纯东移仍可能卡),西退 1.2s(≈0.45m 回平台)后重新东移,
            # 最多 2 次;west 回退 vx=-0.40 vy=0 wz=0 纯退,无对角。
            if (self.east_align_east_backoff_until is not None and
                    now < self.east_align_east_backoff_until):
                vx=-self.landing_recenter_speed
                # R54:北退必须带 south 分量 —— 批次7 run4 实证纯西退
                # 治标不治本(backoff done y=2.07 仍北,resume east 后
                # roll 仍 >0.5 立即再触发,配额耗尽 30s 超时;对比 run2
                # backoff done y=1.93 resume 成功)。y>1.88 时边西退边
                # 南修,把 y 拉回 east 阶段 y 伺服的可靠工作带再 resume。
                vy=(-self.segment_turn_lateral_speed
                    if self.truth_pose[1] > 1.88 else 0.0)
                wz=0.0
                self._publish_truth_world_command(vx, vy, wz)
                self._record_trace()
                return
            elif (self.east_align_east_backoff_until is not None and
                    now >= self.east_align_east_backoff_until):
                if self.truth_pose[1] > 1.92 and \
                        self.east_align_east_backoff_ext < 4:
                    # 南修尚未到位(y 仍北),延长 backoff 继续 south
                    self.east_align_east_backoff_ext+=1
                    self.east_align_east_backoff_until=now+0.6
                    self.east_align_east_stall_at=None
                    rospy.logwarn('East-align backoff y=%.2f still north; '
                                  'extending south correction (%d/4).',
                                  self.truth_pose[1],
                                  self.east_align_east_backoff_ext)
                    return
                self.east_align_east_backoff_until=None
                self.east_align_east_backoff_ext=0
                self.east_align_east_stall_at=None
                self.east_align_east_stall_pos=self.truth_pose[:2]
                rospy.loginfo('East-align backoff done at (%.2f, %.2f); '
                              'resuming east.', self.truth_pose[0],
                              self.truth_pose[1])
            # R65(2026-08-30 批次21 run4):z 陷落深回退 —— run4 实证
            # EAST_ALIGN z 掉穿 tread_z(2.75)后东移失力卡死(130s 超时):
            # 北退 retry 3/3 耗尽 + 南修 extend 4/4 拉不回 y(过渡带步态
            # 失力,必须先回平台再修 y)。z≤tread_z 且 retry 耗尽时深回退
            # 到平台(x≤-3.45,z 恢复 2.95)再 resume;成功轮 z 最低 2.85
            # 不触发,零回归。
            # R67(批次24 run3):深回退补 y 分支 —— run3 实证 z=2.85 完全
            # 正常(成功轮范围)但 y 北漂 1.96-1.99,北退南修 extend 4/4 拉
            # 不回(过渡带步态失力),resume 后 y>1.90 触发 y-priority
            # (vx=0 纯南修)反而北漂,10s 停滞 → 30s 超时。北退 1 次 +
            # 南修 4/4 耗尽仍 y>1.92 = 原地修不回,深回退平台(x≤-3.45
            # 平台平地 y 伺服有效,成功轮模式)再 resume。
            if (self.east_align_force_deep or
                ((self.east_align_east_retries >= 3 and
                  self.truth_pose[2] <= self.east_align_tread_z) or
                 (self.east_align_east_retries >= 1 and
                  self.east_align_east_backoff_ext >= 4 and
                  self.truth_pose[1] > 1.92))):
                if self.east_align_deep_backoff_until is None:
                    if self.east_align_deep_backoff_count >= 2:
                        # R70:deep 2 次后原逻辑每 tick 重置 stall_at(位移门
                        # 永久失效,与 retry 耗尽同死锁)。改为静止累计 10s
                        # → 重置 retries 再试一轮(平台平地 backoff 有效)。
                        if self.east_align_east_stall_at is None:
                            self.east_align_east_stall_at=now
                        elif now-self.east_align_east_stall_at >= 10.0:
                            self.east_align_east_retries=0
                            self.east_align_force_deep=False
                            self.east_align_east_backoff_until=now+1.2
                            self.east_align_east_stall_at=None
                            rospy.logwarn(
                                'East-align still frozen after deep '
                                'backoff (z=%.2f y=%.2f); resetting '
                                'retries.', self.truth_pose[2],
                                self.truth_pose[1])
                    else:
                        self.east_align_deep_backoff_count+=1
                        self.east_align_deep_backoff_until=now+10.0
                        self.east_align_east_stall_at=None
                        rospy.logwarn(
                            'East-align stuck (z=%.2f y=%.2f x=%.2f, '
                            'retries exhausted/south ineffective); deep '
                            'backing off to platform (%d/2).',
                            self.truth_pose[2], self.truth_pose[0],
                            self.truth_pose[1],
                            self.east_align_deep_backoff_count)
                if (self.east_align_deep_backoff_until is not None and
                        now < self.east_align_deep_backoff_until and
                        (self.truth_pose[2] < 2.85 or
                         self.truth_pose[0] > -3.45)):
                    vx=-self.landing_recenter_speed
                    vy=(-self.segment_turn_lateral_speed
                        if self.truth_pose[1] > 1.88 else 0.0)
                    self._publish_truth_world_command(vx, vy, 0.0)
                    self._record_trace()
                    return
                self.east_align_deep_backoff_until=None
                self.east_align_east_stall_at=None
                self.east_align_east_stall_pos=self.truth_pose[:2]
                self.east_align_force_deep=False
                rospy.loginfo('East-align deep backoff done (x=%.2f '
                              'y=%.2f z=%.2f); resuming east.',
                              self.truth_pose[0], self.truth_pose[1],
                              self.truth_pose[2])
            if self.east_align_east_stall_at is None:
                self.east_align_east_stall_at=now
                self.east_align_east_stall_pos=self.truth_pose[:2]
            # R69(2026-08-31 批次28 run4):位移阈值 0.02→0.06 —— run4 实证
            # 东移冻结时 x 在 -3.31±0.04 原地微晃(滑移型),3s 窗口总位移
            # 0.018-0.068 周期内恰好 >0.02 重置 stall 门,35s 无净进展;
            # z 恒 2.85+ 正常又绕过 R64 z 门/北退 roll 门,45s 硬超时。
            # 0.06 覆盖微晃幅度;正常东移 3s 位移远大于 0.06,零误伤;
            # 触发后 backoff 在平台平地一定有效(非楔死型)。
            elif (math.hypot(self.truth_pose[0]-self.east_align_east_stall_pos[0],
                             self.truth_pose[1]-self.east_align_east_stall_pos[1])
                    < 0.06):
                if now-self.east_align_east_stall_at >= 3.0:
                    if self.east_align_east_retries < 3:
                        self.east_align_east_retries+=1
                        self.east_align_east_backoff_until=now+1.2
                        self.east_align_east_backoff_ext=0
                        self.east_align_east_stall_at=None
                        rospy.logwarn(
                            'East-align east stalled at (%.2f, %.2f) '
                            'z=%.2f; backing off west, retry %d/3.',
                            self.truth_pose[0], self.truth_pose[1],
                            self.truth_pose[2], self.east_align_east_retries)
                    else:
                        # R70(2026-08-31 批次32 run2):原逻辑每 tick 重置
                        # stall_at → 位移门永久失效,完全静止(位移≈0)死锁
                        # 至 45s 超时(run2 实证 93-115s x/y/z/yaw 全静止,
                        # backoff 从未触发)。改为静止累计 10s → 重置
                        # retries 再试一轮(平台平地 backoff 历史有效);
                        # 轮上限 2 后强制深回退(下方 east_align_force_deep,
                        # 平台 x≤-3.45 平地 y 伺服有效)。
                        if self.east_align_east_stall_at is None:
                            self.east_align_east_stall_at=now
                        elif now-self.east_align_east_stall_at >= 10.0:
                            if self.east_align_retry_rounds < 2:
                                self.east_align_retry_rounds+=1
                                self.east_align_east_retries=0
                                self.east_align_east_backoff_until=now+1.2
                                self.east_align_east_backoff_ext=0
                                self.east_align_east_stall_at=None
                                rospy.logwarn(
                                    'East-align still frozen (z=%.2f '
                                    'y=%.2f); resetting retries, '
                                    'round %d/2.',
                                    self.truth_pose[2], self.truth_pose[1],
                                    self.east_align_retry_rounds)
                            else:
                                # 2 轮(≥6 次 backoff)仍冻结 → 强制深回退
                                self.east_align_force_deep=True
                                self.east_align_east_stall_at=now
            else:
                self.east_align_east_stall_at=now
                self.east_align_east_stall_pos=self.truth_pose[:2]
                self.east_align_force_deep=False
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
        # 速度整形:记录本 flight 起点 z(按 flight 本地 drop 分档速度);
        # 倾角降速窗口逐 flight 重置。
        self.flight_start_z=(self.truth_pose[2]
                             if self.truth_pose is not None else None)
        self.flight_tilt_slow_until=None
        self.flight_severe_tilt_until=None
        self.flight_stall_since=None
        self.flight_stall_window_start_z=None
        self.flight_stall_window_start_y=None
        self.flight_stall_recoveries=0
        self.flight_pause_until=None
        self.flight_ramp_until=None
        self.flight_backoff_until=None
        self.flight_yaw_recover_until=None
        self.flight_descent_min_z=(self.truth_pose[2]
                                   if self.truth_pose is not None else None)
        self.fall_window_until=time.monotonic()+self.fall_window_seconds
        self.fall_window_start_z=(self.truth_pose[2]
                                  if self.truth_pose is not None else None)
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

    # ------------------------------------------------------------------ #
    # tick 主循环
    # ------------------------------------------------------------------ #
    def tick(self, _):
        desired_pause=self.phase in (
            'STAIR_DESCENT_ENTRY_GUIDE', 'STAIR_DESCENT_PRE_ALIGN',
            'STAIR_DESCENT_POLICY_LOADING', 'STAIR_DESCENT_POLICY_WARMUP',
            'STAIR_DESCENT_STAND', 'STAIR_DESCENT_EAST_ALIGN',
            'STAIR_DESCENT_FLIGHT_B', 'STAIR_DESCENT_TURN',
            'STAIR_DESCENT_FLIGHT_A', 'STAIR_DESCENT_SEGMENT_TURN')
        if desired_pause != self._pause_active:
            self._pause_active=desired_pause
            self._pause_pub.publish(Bool(data=desired_pause))

        if self.phase == 'WAIT_F3':
            if self.started is None and \
                    time.monotonic() > self._boot_monotonic + \
                    self.trigger_timeout:
                self._terminal('STAIR_DESCENT_TRIGGER_TIMEOUT',
                               'stair_descent_trigger_timeout', error=True)
            # 触发消息到达前此 tick 无事可做,50Hz 空转在满载仿真机上
            # 实测烧 ~75% CPU(round 3);降频等待,代价是最多延迟 0.5s
            # 感知触发,对后续 STAND/ENTRY_GUIDE 无影响。
            time.sleep(0.5)
            return
        if self.truth_pose is None:
            return

        if self.phase == 'STAIR_DESCENT_ENTRY_GUIDE':
            self._entry_guide_tick()
            return
        if self.phase == 'STAIR_DESCENT_PRE_ALIGN':
            self._pre_align_tick()
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
        self._record_trace()

    # ------------------------------------------------------------------ #
    # 入口引导(F3 走廊口 → F3 landing 中心,朝向 flight_b 下行方向)
    # ------------------------------------------------------------------ #
    def _truth_entry_reenter_route(self):
        """R41b:handoff 起点在房间内时先经门洞重入走廊中心线。

        batch7 RUN3(STAIR_ENTRY_NOT_REACHED,穿墙卡死)与 batch8 RUN2
        (ENTRY_TIMEOUT,房间内冻结)都源于探索异常收尾(TIME_LIMIT/
        GOAL_FAILURE)把机器人留在房间内,下行直线引导从房间内穿墙。
        正常 EXIT_HANDOFF 终点在走廊中心线(x∈[-1.1,1.1]),不触发。
        布局来自 offline_truth_layout_metadata(floor_index = start_floor-1)。
        """
        if self.truth_pose is None or self.truth_offline_layout is None:
            return []
        floor_index=self.start_floor_number-1
        rooms=None
        for floor in self.truth_offline_layout.get('floors', []):
            if int(floor.get('floor_index', -1)) == floor_index:
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
                    'Descent handoff pose (%.2f, %.2f) is inside a room; '
                    're-entering corridor via door x=%.2f y=%.2f.',
                    px, py, door_x, door_y)
                return [room_side, corridor_centre]
        return []

    def _entry_guide_tick(self):
        if self.truth_pose is None:
            return
        # R44:入口起点姿态异常(roll≈±π,batch8 RUN2 实证探索收尾后姿态
        # 奇异,任何命令无效冻结 90s)→ 2s 宽限后快速失败,不空耗入口
        # 超时。正常入口引导 roll 恒 <0.3。
        if self.truth_attitude is not None:
            roll_abs=abs(self.truth_attitude[0])
            if roll_abs > 1.2:
                if self.truth_entry_roll_bad_at is None:
                    self.truth_entry_roll_bad_at=time.monotonic()
                elif time.monotonic()-self.truth_entry_roll_bad_at >= 2.0:
                    self._terminal('STAIR_DESCENT_ENTRY_ATTITUDE_FAILED',
                                   'stair_descent_entry_attitude_failed',
                                   error=True)
                    return
            else:
                self.truth_entry_roll_bad_at=None
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
        # R41b:重入走廊期间覆盖 corridor/side 路点,直走到下一个重入点。
        if self.truth_reenter_points:
            target=self.truth_reenter_points[0]
            self.truth_using_corridor_return=False
        elif self.truth_entry_stage == 'corridor':
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
            if self.truth_reenter_points:
                self.truth_reenter_points.pop(0)
                self.truth_entry_best_distance=None
                self.truth_entry_watchdog_anchor_distance=None
                self.truth_entry_deadline=(
                    time.monotonic()+self.truth_entry_timeout)
                if self.truth_reenter_points:
                    rospy.loginfo('Descent re-entry point reached; '
                                  'heading to the corridor centreline.')
                else:
                    rospy.loginfo('Descent corridor re-entry complete; '
                                  'continuing the normal corridor approach.')
                    self.truth_entry_stage='corridor'
                    self.truth_entry_stage_switches+=1
                self._record_trace()
                return
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
                # R69:重置转体 watchdog 状态
                self.pre_align_turn_started_at=None
                self.pre_align_yaw_anchor=None
                self.pre_align_recover_until=None
                self.pre_align_recoveries=0
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
            self._enter_policy_loading()
            self._record_trace(force=True)
            return
        # R69(2026-08-31 批次28 run2):纯转体 (0,0,wz) 冻结 900s
        # (命令每秒发布但 yaw 完全不动,无超时 → wrapper NO_TRACE)。
        # stair 策略的转身 = "body 横向侧移 + 同向 yaw"(SEGMENT_TURN
        # R8j 修复同源,注释 2818-2820),纯转体是 R8j 缺陷模式(run1
        # 纯靠侧移惯性碰巧通过)。改为 SEGMENT_TURN 同款:侧移 0.40
        # (world 投影)+ 2.0×error 强增益 + 头 2s 慢启动 + x 保持对中。
        if self.pre_align_turn_started_at is None:
            self.pre_align_turn_started_at=time.monotonic()
            self.pre_align_yaw_anchor=error
        # 转体 watchdog:20s 窗口 yaw 总进展 <0.1 rad = 步态冻结 →
        # 零命令重植 1.5s;重植后再冻结(累计 2 次)快速失败,避免
        # NO_TRACE 900s 死等(run2 实测 15.6min 无任何检测触发)。
        if self.pre_align_recover_until is None:
            if (time.monotonic()-self.pre_align_turn_started_at >= 20.0
                    and abs(error-self.pre_align_yaw_anchor) < 0.1):
                if self.pre_align_recoveries < 2:
                    self.pre_align_recoveries+=1
                    self.pre_align_recover_until=time.monotonic()+1.5
                    self.pre_align_turn_started_at=None
                    rospy.logwarn('Pre-align turn frozen (yaw err %.2f '
                                  'rad unmoving); re-planting gait '
                                  '(recovery %d/2).', error,
                                  self.pre_align_recoveries)
                else:
                    self._terminal('STAIR_DESCENT_PRE_ALIGN_TIMEOUT',
                                   'stair_descent_pre_align_timeout',
                                   error=True)
                    return
        else:
            if time.monotonic() < self.pre_align_recover_until:
                # 零命令重植步态(SEGMENT_TURN landing_slip 同款)
                self.cmd.publish(Twist())
                self.hold_rl()
                self._record_trace()
                return
            self.pre_align_recover_until=None
            self.pre_align_yaw_anchor=error
        yaw_rate=self._segment_turn_strong_rate(error)
        # R37 慢启动(SEGMENT_TURN 同款):头 2s 0.24→满速,大步态起步
        # 即滑落(修复 J 实证 0.10 起步转不动,0.24 是死区上缘)。
        ramp_since=time.monotonic()-self.pre_align_turn_started_at
        if 0.0 <= ramp_since < 2.0:
            cap=(0.24+(self.segment_turn_yaw_rate-0.24)*ramp_since/2.0)
            yaw_rate=max(-cap, min(cap, yaw_rate))
        lat=(self.segment_turn_lateral_speed
             if yaw_rate >= 0.0 else -self.segment_turn_lateral_speed)
        # world 投影 = R(yaw)·(0, lat),投影回 body 恒为纯侧移
        vx=-lat*math.sin(self.truth_pose[3])
        vy=lat*math.cos(self.truth_pose[3])
        # x 保持对中(转体侧移带 x 漂移,SEGMENT_TURN 同款)
        x_err=center_x-self.truth_pose[0]
        vx+=max(-self.landing_recenter_speed,
                min(self.landing_recenter_speed,
                    self.landing_position_gain*x_err))
        self._publish_truth_world_command(vx, vy, yaw_rate)
        self._record_trace()

    def _policy_loading_tick(self):
        if self.policy_loaded:
            self.phase='STAIR_DESCENT_POLICY_WARMUP'
            self.state.publish(String(data=self.phase))
            self.policy_warmup_until=(time.monotonic() +
                                      self.policy_warmup_seconds)
            rospy.loginfo('Descent stair policy loaded; warming up %.1f s.',
                          self.policy_warmup_seconds)
        else:
            self.cmd.publish(Twist())
            self.hold_rl()
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

    def _scheduled_flight_speed(self):
        """按本 flight 垂直进度 + 倾角/误差修正计算当前轴速上限。

        迁移自 scanplanner 方案(2026-09-01):descent_speed 是巡航目标
        (0.80),但实际命令由本地 drop 分档整形 —— 起步 0.55(接触首级
        台阶前降速),drop 0.18-0.45m 线性爬升,末端 drop≥0.95m 收 0.60;
        倾角 ≥0.48/0.60 rad 时按 hold 时长压速(正常下梯姿态约 0.45)。
        误差修正降速由 _flight_control 在调用后按 correction 状态再压。
        """
        speed=max(0.0, self.descent_speed)
        if self.truth_pose is not None and self.flight_start_z is not None:
            local_drop=max(0.0, self.flight_start_z-self.truth_pose[2])
            if local_drop <= self.flight_initial_hold_drop:
                speed=min(speed, self.flight_initial_speed)
            elif local_drop < self.flight_initial_ramp_end_drop:
                span=max(1e-6, self.flight_initial_ramp_end_drop-
                         self.flight_initial_hold_drop)
                progress=(local_drop-self.flight_initial_hold_drop)/span
                ramp_speed=(self.flight_initial_speed +
                            (self.descent_speed-self.flight_initial_speed)*
                            progress)
                speed=min(speed, ramp_speed)
            if local_drop >= self.flight_end_slowdown_drop:
                speed=min(speed, self.flight_end_speed)

        now=time.monotonic()
        if self.truth_attitude is not None:
            tilt=max(abs(self.truth_attitude[0]),
                     abs(self.truth_attitude[1]))
            if tilt >= self.flight_severe_tilt_slowdown:
                self.flight_severe_tilt_until=max(
                    self.flight_severe_tilt_until or now,
                    now+self.flight_severe_tilt_hold)
                rospy.logwarn_throttle(
                    1.0, 'Descent severe tilt %.3f rad: emergency speed cap '
                    '%.2f m/s.', tilt, self.flight_severe_tilt_speed)
            elif tilt >= self.flight_tilt_slowdown:
                self.flight_tilt_slow_until=max(
                    self.flight_tilt_slow_until or now,
                    now+self.flight_tilt_hold)
                rospy.logwarn_throttle(
                    1.0, 'Descent tilt %.3f rad: capping flight speed at '
                    '%.2f m/s.', tilt, self.flight_tilt_speed)
        if (self.flight_tilt_slow_until is not None and
                now < self.flight_tilt_slow_until):
            speed=min(speed, self.flight_tilt_speed)
        elif (self.flight_tilt_slow_until is not None and
              now >= self.flight_tilt_slow_until):
            self.flight_tilt_slow_until=None
        if (self.flight_severe_tilt_until is not None and
                now < self.flight_severe_tilt_until):
            speed=min(speed, self.flight_severe_tilt_speed)
        elif (self.flight_severe_tilt_until is not None and
              now >= self.flight_severe_tilt_until):
            self.flight_severe_tilt_until=None
        return speed

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
        # 迁移自 scanplanner 方案(2026-09-01):中心/航向误差超阈值时
        # 升级 yaw 率上限并压低轴速 —— 高速下梯时漂移在硬守卫前就
        # 提前减速修正,而不是等 side-slip 守卫(pause 2s)打断节奏。
        correction_active=(
            abs(center_error) >= self.flight_center_correction_error)
        heading_correction_active=(
            abs(heading_error) >= self.flight_heading_correction_error)
        yaw_limit=(max(self.max_yaw_rate,
                       self.flight_center_correction_yaw_rate)
                   if correction_active else self.max_yaw_rate)
        if heading_correction_active:
            yaw_limit=max(yaw_limit,
                          self.flight_heading_correction_yaw_rate)
        yaw_magnitude=min(yaw_limit,
                          abs(self.heading_gain*heading_error))
        yaw_rate=(math.copysign(yaw_magnitude, heading_error)
                  if abs(heading_error) > 1e-6 else 0.0)
        # 纯轴:世界系命令沿当前航向(正向朝下梯方向,后向背身)。
        # 轴速 = 速度整形(进度/倾角分档)后,再按修正状态压速。
        axis_speed=self._scheduled_flight_speed()
        if correction_active:
            axis_speed=min(axis_speed, self.flight_center_correction_speed)
        if heading_correction_active:
            axis_speed=min(axis_speed, self.flight_heading_correction_speed)
        body_axis=(-axis_speed if self.backward_mode else axis_speed)
        vx=body_axis*math.cos(self.truth_pose[3])
        vy=body_axis*math.sin(self.truth_pose[3])
        return vx, vy, yaw_rate, center_error, heading_error

    def _flight_tick(self, which):
        if self.truth_pose is None or self.descent_start_z is None:
            return
        now=time.monotonic()
        # R52:FLIGHT 段翻倒护栏 —— |roll|>0.9 rad(52°)瞬时判翻倒。
        # 正常下梯 roll 恒 <0.3;批次5 run3 段2 FLIGHT_B 接棒时 roll
        # 已 -1.1(EAST_ALIGN 滑落漏检兜底),旧逻辑 29 次 stall 恢复
        # 挣扎 185s 才 abort。EAST_ALIGN 的 R50/R52 检测只在 east_align
        # 阶段运行,交棒后由本护栏兜底,快速终止暴露失败。
        if abs(self.truth_attitude[0]) > 0.9:
            self._terminal('STAIR_DESCENT_FLIGHT_TIP_OVER',
                           'stair_descent_flight_tip_over', error=True)
            return
        elapsed=now-self.flight_started_at
        drop=self._drop()
        if self.flight_descent_min_z is None:
            self.flight_descent_min_z=self.truth_pose[2]
        else:
            self.flight_descent_min_z=min(self.flight_descent_min_z,
                                          self.truth_pose[2])
        # ---- 坠落守卫:短窗内 z 骤降(自然下梯 ~0.17 m/s) ----
        if self.fall_window_until is None or now >= self.fall_window_until:
            self.fall_window_until=now+self.fall_window_seconds
            self.fall_window_start_z=self.truth_pose[2]
        elif self.fall_window_start_z is not None and \
                self.fall_window_start_z-self.truth_pose[2] > self.fall_drop:
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
            if drop >= self.flight_b_descent_drop and \
                    self.truth_pose[1] >= self.turn_start_y:
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
                self._segment_complete()
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
                z_drop=self.flight_stall_window_start_z-self.truth_pose[2]
                y_adv=self._flight_y_advance_from(
                    self.flight_stall_window_start_y)
                self.flight_stall_since=now
                self.flight_stall_window_start_z=self.truth_pose[2]
                self.flight_stall_window_start_y=self.truth_pose[1]
                if z_drop < self.stall_z_progress and \
                        y_adv < self.stall_y_progress:
                    self.flight_stall_recoveries+=1
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
        turn_error=unbounded
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
        minimum=min(maximum, max(0.0, self.landing_minimum_yaw_rate*0.5))
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

    def _segment_turn_side_vy(self, err):
        """SEGMENT_TURN y 向侧移命令:比例伺服 + 步态死区下限。

        stair 策略对 <~0.24 m/s 的 body 侧移命令基本无视(实测死区
        ≈0.24,TURN/east_align fixT 实证有效侧移率 0.40)。比例增益
        0.90 下 |y_err|<0.27 m 的输出全部落入死区 → 位置冻结
        (R46 batch18 实证:落点 y=2.165-2.181 仅低于门 2.20,settle
        推 0.10 / align 推 vy≈0.02-0.03 全无效,3/3 轮
        SEGMENT_TURN_TIMEOUT)。误差 >0.01 m 时强制下限 side_speed
        (0.40,east_align fixT 同款 min-rate 地板);误差收敛后比例
        输出归零,由门判定(y≥2.20)放行,不会抖动。
        """
        vy=self.landing_position_gain*err
        if err >= 0.01:
            vy=max(vy, self.segment_turn_side_speed)
        elif err <= -0.01:
            vy=min(vy, -self.segment_turn_side_speed)
        return max(-self.segment_turn_side_speed,
                   min(self.segment_turn_side_speed, vy))

    def _segment_turn_tick(self):
        if self.truth_pose is None:
            return
        now=time.monotonic()
        if now-self.segment_turn_started_at >= self.segment_turn_timeout:
            self._terminal('STAIR_DESCENT_SEGMENT_TURN_TIMEOUT',
                           'stair_descent_segment_turn_timeout', error=True)
            return
        # R68(2026-08-30 批次25 run4):SEGMENT_TURN 翻倒检测 —— run4
        # 实证转体中途侧翻(roll=-1.519,z 掉到 2.705 是翻倒伴随而非
        # 步态失力),被下方 R67 z 门当陷落交棒 FLIGHT_B → 15ms 后
        # TIP_OVER(已翻交棒无效,白耗 3s)。R50/R52 双重阈值同款
        # (EAST_ALIGN 段已证有效):|roll|>0.9 瞬时 或 |roll|>0.5 持续
        # 0.8s → 干净报错,不浪费恢复循环。
        if abs(self.truth_attitude[0]) > 0.9:
            self._terminal('STAIR_DESCENT_SEGMENT_TURN_TIP_OVER',
                           'stair_descent_segment_turn_tip_over',
                           error=True)
            return
        if abs(self.truth_attitude[0]) > 0.5:
            if self.segment_turn_roll_bad_at is None:
                self.segment_turn_roll_bad_at=now
            if now-self.segment_turn_roll_bad_at >= 0.8:
                self._terminal('STAIR_DESCENT_SEGMENT_TURN_TIP_OVER',
                               'stair_descent_segment_turn_tip_over',
                               error=True)
                return
        else:
            self.segment_turn_roll_bad_at=None
        # R67(2026-08-30 批次24):绝对 z 兜底交棒(每 tick 顶层,覆盖
        # 推进/slip/backoff 全分支)—— 批次24 实证相对 z_drop 门三种
        # 失效场景:(a) run4 降落即低(SEGMENT_TURN 起点 z 就 2.684,
        # z_drop 恒 0,卡推进分支 1002s);(b) run5 掉落恰在 re-planted
        # 判定期间,landing_z0 被写低 2.737 后 z_drop 永久 0;(c) run2
        # 掉到一层 z=0.388,yaw 乱(>0.6)被 yaw 门拦截。z≤tread_z(2.75)
        # 是步态失力临界(成功轮转体全程 z≥2.85),与起点无关,位置
        # 冻结/滑移均不影响判定:
        #   a) z≤2.50 瞬时 → 掉一层/深陷,立即交棒(不管 yaw/z_drop)
        #   b) z≤2.75 持续 3s → 陷落型交棒(R68:交棒前 yaw_err>0.6
        #      时先原地转体对齐,防转体未完成歪起步;run4 的
        #      yaw=-1.755 已对齐差 0.18,此门不改变其翻倒结果)
        if self.truth_pose[2] <= self.segment_turn_low_z_instant:
            rospy.logwarn('Descent segment-turn z=%.3f critically low; '
                          'handing off to flight-b.', self.truth_pose[2])
            self._begin_flight('b')
            self._record_trace()
            return
        if self.truth_pose[2] <= self.east_align_tread_z:
            if self.segment_turn_low_z_at is None:
                self.segment_turn_low_z_at=now
            if now-self.segment_turn_low_z_at >= \
                    self.segment_turn_low_z_hold:
                yaw_target=(self._flight_heading(
                    self.truth_flight_b_heading)
                    if self.truth_flight_b_heading is not None
                    else math.pi/2.0)
                yaw_err=math.atan2(
                    math.sin(yaw_target-self.truth_pose[3]),
                    math.cos(yaw_target-self.truth_pose[3]))
                if abs(yaw_err) > 0.6:
                    if self.segment_turn_prep_yaw_until is None:
                        self.segment_turn_prep_yaw_until=now+5.0
                        rospy.logwarn('Descent segment-turn trapped low '
                                      'z=%.3f (yaw err %.2f); aligning '
                                      'yaw before handoff.',
                                      self.truth_pose[2], yaw_err)
                    if now < self.segment_turn_prep_yaw_until:
                        wz=max(-self.segment_turn_yaw_rate,
                               min(self.segment_turn_yaw_rate,
                                   2.0*yaw_err))
                        # R69:同 PRE_ALIGN —— 纯转体是 R8j 缺陷模式,
                        # 带侧移转体(SEGMENT_TURN 同款)成功率更高;窗口
                        # 结束仍歪由 R68 交棒兜底,零回归。
                        lat=(self.segment_turn_lateral_speed
                             if wz >= 0.0 else
                             -self.segment_turn_lateral_speed)
                        vx=-lat*math.sin(self.truth_pose[3])
                        vy=lat*math.cos(self.truth_pose[3])
                        self._publish_truth_world_command(vx, vy, wz)
                        self._record_trace()
                        return
                self.segment_turn_prep_yaw_until=None
                rospy.logwarn('Descent segment-turn trapped low z=%.3f '
                              'for %.1fs; handing off to flight-b.',
                              self.truth_pose[2],
                              now-self.segment_turn_low_z_at)
                self._begin_flight('b')
                self._record_trace()
                return
        else:
            self.segment_turn_low_z_at=None
        # ---- R37 落步稳定:FLIGHT_A 末端着地后先零命令 settle,等步态
        #      站稳再掉头(R17 实证着地打滑导致 align/转体冻结) ----
        if self.landing_settle_remaining > 0.0:
            # R38/R39 防陷:落点 y<2.15 时后腿在平台南缘(tread 过渡带)
            # 上方,零命令落步会滑陷(batch6 RUN1 实证 settle 0.4s 内
            # z 3.04→2.94、y 2.05→1.97,深位后北推无效 → 死锁;RUN3
            # 实证门槛 2.10 仍不够 —— 落点 2.135 未触发,落步 1s 缓滑
            # y 2.135→2.07,后推回拉锯 40s 超时)。settle 期间给小幅
            # world +y 推(未到 guard_recenter 2.15 就推),让后腿落在
            # 平台平坦区(质心≥~2.15)再进 align。
            if self.truth_pose[1] < self.segment_turn_y_guard_recenter:
                push=min(self.segment_turn_side_speed,
                         max(0.10, (self.segment_turn_y_guard_recenter -
                                    self.truth_pose[1])*1.5))
                self._publish_truth_world_command(0.0, push, 0.0)
                self._record_trace()
            else:
                self.cmd.publish(Twist())
                self._record_trace()
            self.hold_rl()
            self.landing_settle_remaining-=0.02
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
                # R57(2026-08-29 批次10 run5):回退方向自适应 —— R17 固定
                # east 只在"flight_a 末端西侧跌落"场景成立;掉头东滑后
                # (run5 实证 x=-3.40 在 x_keep=-3.70 东侧 0.30)固定 east
                # 回退 0.55m 把 x 推到 -2.85 出平台,重植后 align 推回
                # 距离更大且 y 仍 <2.20(R43 推 x 必滑)→ 6 次 recovery
                # 全耗尽仍超时。改为向 x_keep 方向回退;同时 y 低于
                # 安全带(2.20)时加北推 —— y 回 2.20+ 后 align 推 x 才
                # 不滑(R54 east backoff 的 south 分量同款思路)。
                backoff_dir=(1.0 if self.truth_pose[0] <
                             self.segment_turn_x_keep else -1.0)
                # R59(2026-08-29 批次11 run5):backoff y 分量对称化 ——
                # R57 只推 north(y<2.20 时),run5 实证重植楔死在北侧
                # y=2.59(align 拉回 2.40 推不动)→ 6 次 recovery 全耗
                # 尽;y>2.40(y_high)时 backoff 需推 south 回掉头带。
                backoff_vy=0.0
                if self.truth_pose[1] < self.segment_turn_y_guard_recenter:
                    backoff_vy=self.segment_turn_side_speed
                elif self.truth_pose[1] > self.segment_turn_y_high:
                    backoff_vy=-self.segment_turn_side_speed
                self.landing_slip_backoff_dir=backoff_dir
                self.landing_slip_backoff_vy=backoff_vy
                # R61:记录 backoff 起点(结束块据此判定推不动升级)
                self.segment_turn_backoff_origin=self.truth_pose[:2]
                # R62:升级距离从参数起算,升级时累乘(见结束块)
                self.segment_turn_backoff_dist=self.landing_slip_backoff_distance
                self.landing_slip_backoff_until=(now +
                    self.segment_turn_backoff_dist /
                    max(0.05, self.landing_slip_backoff_speed))
                rospy.logwarn(
                    'Descent segment-turn slip recovery: backing %s '
                    '(%.2f m%s) before retrying.',
                    'east' if backoff_dir > 0 else 'west',
                    self.landing_slip_backoff_distance,
                    (' + north' if backoff_vy > 0 else
                     ' + south' if backoff_vy < 0 else ''))
            if now < self.landing_slip_backoff_until:
                self._publish_truth_world_command(
                    self.landing_slip_backoff_dir*self.landing_slip_backoff_speed,
                    self.landing_slip_backoff_vy, 0.0)
                self._record_trace()
                return
            # R61(2026-08-29 批次14 run5):backoff 失效检测 + y 带外持续
            # 修正 —— 实证 yaw stall 恢复重植到 y=3.016(带外北侧 0.6m),
            # align 步态带外失力冻结,backoff west+south 0.55m 推完位置
            # 纹丝不动,原地重植死循环 6/6 耗尽(批次11 run2/run5 楔死
            # 同款复发)。backoff 结束时:
            #   a) 位移 <0.02m = 推不动 → 升级距离 ×1.5 再推(最多 2 次
            #      升级:0.55→0.83→1.24m,方向/vy 不变),仍不动保底 resume
            #   b) y 仍在掉头带外(带 [1.70,2.40] ±0.1 裕量)→ 不 resume,
            #      延长 backoff 继续 y 修正(最多 4 次 ×0.6s,同 EAST_ALIGN
            #      extend)—— y 回带内 align 步态才有力
            moved=math.hypot(
                self.truth_pose[0]-self.segment_turn_backoff_origin[0],
                self.truth_pose[1]-self.segment_turn_backoff_origin[1]) \
                if self.segment_turn_backoff_origin is not None else 1.0
            # R64(2026-08-30 批次20 run1):楔死交棒独立 z 陷落门 ——
            # run1 实证楔死是"z 陷 tread 过渡带 2.66(< tread_z 2.75)
            # 150s+ 缓慢东滑"(moved 每 tick 0.005-0.05m,累计 ≥0.02)→
            # 永远进不了下方 moved<0.02 的 R61 升级/R62 交棒分支(批次
            # 15 run5 是完全冻结型 moved 0.000 已覆盖;run1 是滑移型,
            # 位移门/y 带外门全绕过,6/6 recovery 耗尽超时)。z 陷落 +
            # yaw 接近下行朝向是楔死最强证据:不依赖 moved 直接交棒
            # FLIGHT_B(飞行步态 body 轴下降,不依赖世界系平移,能救出)。
            yaw_target=(self._flight_heading(self.truth_flight_b_heading)
                        if self.truth_flight_b_heading is not None
                        else math.pi/2.0)
            yaw_err=math.atan2(
                math.sin(yaw_target-self.truth_pose[3]),
                math.cos(yaw_target-self.truth_pose[3]))
            z_drop=(self.landing_z0-self.truth_pose[2]
                    if self.landing_z0 is not None else 0.0)
            if (z_drop >= 0.25 and
                    self.truth_pose[2] <= self.east_align_tread_z and
                    abs(yaw_err) <= 0.6):
                rospy.logwarn('Descent segment-turn trapped (z %.3f, '
                              'drop %.2f, yaw err %.2f, moved %.3f); '
                              'handing off to flight-b.',
                              self.truth_pose[2], z_drop, yaw_err, moved)
                self._begin_flight('b')
                self._record_trace()
                return
            if moved < 0.02:
                if self.segment_turn_backoff_escalations < 2:
                    # R62:升级距离累乘(R61 bug:distance*1.5 每次用参数
                    # 0.55 → 两次都 0.83m,现 0.55→0.83→1.24)
                    self.segment_turn_backoff_escalations+=1
                    self.segment_turn_backoff_dist*=1.5
                    self.landing_slip_backoff_until=(now +
                        self.segment_turn_backoff_dist /
                        max(0.05, self.landing_slip_backoff_speed))
                    rospy.logwarn('Descent segment-turn backoff '
                                  'ineffective (moved %.3f m); '
                                  'escalating to %.2f m (%d/2).',
                                  moved, self.segment_turn_backoff_dist,
                                  self.segment_turn_backoff_escalations)
                    self._record_trace()
                    return
                # R62(批次15 run5):楔死保底交棒 —— escalation 推不动
                # = 世界系命令在台阶过渡带失力(着陆 x=-3.99 楔入 flight_a
                # 北段,align 推 x z 掉 0.15 后步态完全失力,moved 0.000,
                # y 带外修正救不回 x 楔死)。z 已跌(楔入证据)且 yaw 接近
                # 下行朝向时直接交棒 FLIGHT_B(注意:R64 更强条件
                # z_drop≥0.25 且 z≤tread_z 已在上方先检查,此处覆盖
                # z_drop∈[0.10,0.25) 或 z 未跌破 tread_z 的冻结型楔死)。
                if z_drop >= 0.10 and abs(yaw_err) <= 0.6:
                    rospy.logwarn('Descent segment-turn wedge-stuck '
                                  '(moved %.3f m, z drop %.2f, yaw err '
                                  '%.2f); handing off to flight-b.',
                                  moved, z_drop, yaw_err)
                    self._begin_flight('b')
                    self._record_trace()
                    return
            if ((self.truth_pose[1] < 1.60 or
                 self.truth_pose[1] > 2.50) and
                    self.segment_turn_backoff_ext < 4):
                self.segment_turn_backoff_ext+=1
                self.landing_slip_backoff_vy=(
                    self.segment_turn_side_speed
                    if self.truth_pose[1] < 1.60 else
                    -self.segment_turn_side_speed)
                self.landing_slip_backoff_until=now+0.6
                rospy.logwarn('Descent segment-turn backoff y=%.3f '
                              'still out of band; extending y '
                              'correction (%d/4).', self.truth_pose[1],
                              self.segment_turn_backoff_ext)
                self._record_trace()
                return
            self.landing_slip_active=False
            self.landing_slip_backoff_until=None
            self.segment_turn_backoff_origin=None
            self.segment_turn_backoff_escalations=0
            self.segment_turn_backoff_ext=0
            self.segment_turn_backoff_dist=None
            self.segment_turn_started_at=now
            self.segment_turn_stage='align'
            # R67:re-planted 不再把 landing_z0 拉低 —— 批次24 run5 实证
            # 掉落(2.867→2.737)恰在 re-planted 判定期间,landing_z0 被
            # 写为低 z 后 z_drop 永久 0,R64/R62 全失效。max 保持高基线
            # (掉落后 re-planted 不降;正常平台 re-planted z≈2.85 不变)。
            # 新绝对 z 门(上方)不再依赖 z_drop,此为防御性修复。
            self.landing_z0=max(self.landing_z0, self.truth_pose[2]) \
                if self.landing_z0 is not None else self.truth_pose[2]
            self.segment_turn_low_z_at=None
            self.segment_turn_prep_yaw_until=None
            self.landing_yaw_stall_accum=0.0
            self.landing_last_yaw=None
            self.landing_last_yaw_at=None
            self.segment_turn_direction=None
            self.segment_turn_stall_pos=None
            # R58:重植后重置护栏推回无效计时
            self.segment_turn_push_started_at=None
            self.segment_turn_push_start_y=None
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
        # R46(2026-08-26 定稿):position_ok 的 x 门放宽到 max(tol, 0.25),
        # 与 align 放行门(align_x_tol)同值。batch9 RUN1 实证:转体完成
        # 后 x_err=0.248 仅超 tol(运行时 0.20),进阶段 3 修正 —— 阶段 3
        # 的 wz 最小率强制(±0.24)在 error≈0 处符号抖动 → 步态横跳冻结
        # 40s 超时。转体后 x 偏差交给 EAST_ALIGN/FLIGHT_B 头向伺服修正
        # (b6R2 实证 settle 在 z=3.10 推 x 0.36m 成功),转体完成即放行。
        position_ok=(abs(x_err) <= max(self.segment_turn_position_tolerance,
                                       0.25) and
                     self.segment_turn_y_low <= self.truth_pose[1] <=
                     self.segment_turn_y_high)
        # ---- yaw 冻结检测(转体命令已发出但 yaw 几乎不动 = 转体步态打
        #      滑;landing TURN 同款累积检测)。只对 turn 阶段有效:
        #      align 阶段 wz=0 是设计行为,yaw 不动是预期的 —— 若也累积
        #      会在 align 推 x/y 时 6s 必触发恢复循环(batch5 RUN1 与
        #      batch6 RUN1 实证:4 次恢复耗尽 → settle 冻结 → 超时)。
        #      R30 的 align 冻结保护由 45s 阶段超时兜底;R47 起由
        #      segment_turn_stall_pos 位置停滞检测接管(命令发出但位置
        #      不动才触发,不会误伤 wz=0 的 align 设计行为)。 ----
        if self.segment_turn_stage == 'turn' and not (turn_done and position_ok):
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
        else:
            self.landing_yaw_stall_accum=0.0
            self.landing_last_yaw=None
            self.landing_last_yaw_at=None
        if turn_done and position_ok:
            self.segment_turn_done=True
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
            # R43(2026-08-26):align→turn 的 x 门放宽到 0.25(batch7 RUN1
            # 落点 x_err=0.205 仅超 tol 0.20,align 推 x 滑落 2.17→2.02)。
            # turn 阶段有 x 实时伺服(x_hold)对中,转体后 settle 阶段 z 已
            # 升高(b6R2 实证 z=3.10 推 x 0.36m 成功)——x 偏差交给
            # turn/settle 修正,align 不再推 x(y<2.25 时推 x 必滑,而 y
            # 物理推不到 2.25)。
            align_x_tol=max(self.segment_turn_position_tolerance, 0.25)
            if (abs(x_err) <= align_x_tol and
                    self.segment_turn_y_guard_recenter <= self.truth_pose[1] <=
                    self.segment_turn_y_high):
                self.segment_turn_stage='turn'
                self.segment_turn_stall_pos=None
                self.segment_turn_turn_started_at=time.monotonic()
                rospy.loginfo('Segment turn aligned (x err=%.3f m, '
                              'y=%.3f in band [%.2f, %.2f]); '
                              'starting 180 deg turn.',
                              x_err, self.truth_pose[1],
                              self.segment_turn_y_guard_recenter,
                              self.segment_turn_y_high)
            else:
                vy=0.0
                if self.truth_pose[1] < self.segment_turn_y_guard_recenter:
                    # R43:y 未达安全带前纯推 y(vx=0)——推 x 在
                    # y<2.20 时侧向跨步必滑(batch7 RUN1 实证 2.19→2.02)。
                    # R47:推 y 加死区下限(_segment_turn_side_vy,batch18
                    # 实证比例输出 vy≈0.02 落入步态死区 → 冻结 45s)。
                    vx=0.0
                    vy=self._segment_turn_side_vy(
                        self.segment_turn_y_guard_recenter-
                        self.truth_pose[1])
                else:
                    # y 就绪(≥2.20)后推 x 伺服;y 过高时同时拉回。
                    vx=max(-self.landing_recenter_speed,
                           min(self.landing_recenter_speed,
                               self.landing_position_gain*x_err))
                    if abs(vx) < self.landing_minimum_x_speed:
                        vx=math.copysign(self.landing_minimum_x_speed, vx)
                    if self.truth_pose[1] > self.segment_turn_y_high:
                        vy=self._segment_turn_side_vy(
                            self.segment_turn_y_high-self.truth_pose[1])
                # R47:align 位置停滞检测 —— 命令已发出但位置几乎不动 =
                # 步态打滑/楔入 tread 过渡带台阶(R46 batch18 实证:落点
                # y=2.165-2.181 且 z 已落 2.83 时 z_drop 检测失效
                # [landing_z0 在跌落完成后捕获],比例推 y 冻结 45s 超时;
                # 死区下限修掉死区冻结后,残留的物理楔入由本检测兜底 →
                # 落步+回退+重植,与 slip recovery 同款恢复链,次数上限
                # landing_slip_max_attempts)。
                stall_now=(self.truth_pose[0], self.truth_pose[1], now)
                if self.segment_turn_stall_pos is None:
                    self.segment_turn_stall_pos=stall_now
                dx=abs(self.truth_pose[0]-self.segment_turn_stall_pos[0])
                dy=abs(self.truth_pose[1]-self.segment_turn_stall_pos[1])
                if (dx < self.segment_turn_align_stall_tolerance and
                        dy < self.segment_turn_align_stall_tolerance):
                    if (now-self.segment_turn_stall_pos[2] >=
                            self.segment_turn_align_stall_seconds and
                            self.landing_slip_recoveries <
                            self.landing_slip_max_attempts):
                        self.landing_slip_recoveries+=1
                        self.landing_slip_active=True
                        self.landing_slip_until=now+self.landing_slip_pause_seconds
                        self.landing_slip_backoff_until=None
                        self.segment_turn_stall_pos=None
                        rospy.logwarn(
                            'Descent segment-turn align stall: pos frozen '
                            '(%.3f, %.3f) %.1f s while commanding; '
                            're-planting gait (recovery %d/%d).',
                            self.truth_pose[0], self.truth_pose[1],
                            self.segment_turn_align_stall_seconds,
                            self.landing_slip_recoveries,
                            self.landing_slip_max_attempts)
                        self._record_trace()
                        return
                else:
                    self.segment_turn_stall_pos=stall_now
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
                # R45 转体护栏(原 R39 南缘护栏):转体步态侧滑把 y 推低时
                # 暂停转体,world +y 推回安全带再继续。R45(2026-08-26)把
                # 护栏线 2.08→2.15 —— batch8 RUN3 实证转体从 y=2.24 一路
                # 滑到 2.09(z 3.10→2.87 深陷),旧护栏在深陷后才触发推不
                # 回;z 尚高(3.04)时 y 先跌破 2.15,此刻推 y 有效。推回
                # 期间 wz=0,yaw 冻结检测的累积恢复(R36 同款)兜底。
                if self.truth_pose[1] < self.segment_turn_turn_y_guard:
                    # R47:推回加死区下限(原 max(0.0, err*2.0) 在
                    # y=2.14 时仅推 0.12,落入步态死区推不动)。
                    push=self._segment_turn_side_vy(
                        self.segment_turn_y_guard_recenter-
                        self.truth_pose[1])
                    # R58:推回无效检测 —— 转体步态失效时侧移也失力,
                    # 推回期间 y 持续下滑(run2 实证 2.15→1.996)等 z 跌
                    # 触发 recovery 太晚(y 已出 2.05 安全带楔入阶梯)。
                    # 推回 ≥2.5s 净回升 <0.05m 即判定无效提前 recovery。
                    if self.segment_turn_push_started_at is None:
                        self.segment_turn_push_started_at=now
                        self.segment_turn_push_start_y=self.truth_pose[1]
                    elif (self.truth_pose[1] >=
                          self.segment_turn_turn_y_guard or
                          self.truth_pose[1]-self.segment_turn_push_start_y
                          >= self.segment_turn_push_min_recover_m):
                        self.segment_turn_push_started_at=now
                        self.segment_turn_push_start_y=self.truth_pose[1]
                    elif (now-self.segment_turn_push_started_at >=
                          self.segment_turn_push_ineffective_sec and
                          self.landing_slip_recoveries <
                          self.landing_slip_max_attempts):
                        self.landing_slip_recoveries+=1
                        self.landing_slip_active=True
                        self.landing_slip_until=now+self.landing_slip_pause_seconds
                        self.landing_slip_backoff_until=None
                        self.segment_turn_push_started_at=None
                        self.segment_turn_push_start_y=None
                        rospy.logwarn(
                            'Descent segment-turn push ineffective: y=%.3f '
                            'not recovering in %.1f s; re-planting gait '
                            '(recovery %d/%d).',
                            self.truth_pose[1],
                            self.segment_turn_push_ineffective_sec,
                            self.landing_slip_recoveries,
                            self.landing_slip_max_attempts)
                        self._record_trace()
                        return
                    rospy.logwarn('Segment turn y=%.3f below guard %.2f; '
                                  'pushing back to %.2f before continuing.',
                                  self.truth_pose[1],
                                  self.segment_turn_turn_y_guard,
                                  self.segment_turn_y_guard_recenter)
                    # 推回是纯侧移(wz=0),yaw 不动是预期 —— 必须跳过
                    # yaw 冻结检测,否则推回 6s 后必触发 slip recovery
                    # 循环(batch5 RUN1 实证:4 次恢复耗尽 → settle 冻结
                    # → SEGMENT_TURN_TIMEOUT)。
                    self.landing_yaw_stall_accum=0.0
                    self._publish_truth_world_command(0.0, push, 0.0)
                    self._record_trace()
                    return
                # R56:北缘护栏(与上方 R45 南缘护栏对称) —— 掉头转体
                # 步态失效时 y 北漂越界(批次9 run1 实证 2.9→8.5),z 未
                # 跌前推回有效(z 恒 2.91 仍平地)。推回期间跳过 yaw 冻结
                # 检测(同 R45 机制,推回时 yaw 不动是预期)。
                if self.truth_pose[1] > self.segment_turn_turn_y_guard_north:
                    push=-self._segment_turn_side_vy(
                        self.truth_pose[1]-self.segment_turn_y_guard_recenter)
                    # R58:北护栏同款推回无效检测(y 需南移 ≥0.05 才算有效)
                    if self.segment_turn_push_started_at is None:
                        self.segment_turn_push_started_at=now
                        self.segment_turn_push_start_y=self.truth_pose[1]
                    elif (self.truth_pose[1] <=
                          self.segment_turn_turn_y_guard_north or
                          self.segment_turn_push_start_y-self.truth_pose[1]
                          >= self.segment_turn_push_min_recover_m):
                        self.segment_turn_push_started_at=now
                        self.segment_turn_push_start_y=self.truth_pose[1]
                    elif (now-self.segment_turn_push_started_at >=
                          self.segment_turn_push_ineffective_sec and
                          self.landing_slip_recoveries <
                          self.landing_slip_max_attempts):
                        self.landing_slip_recoveries+=1
                        self.landing_slip_active=True
                        self.landing_slip_until=now+self.landing_slip_pause_seconds
                        self.landing_slip_backoff_until=None
                        self.segment_turn_push_started_at=None
                        self.segment_turn_push_start_y=None
                        rospy.logwarn(
                            'Descent segment-turn north push ineffective: '
                            'y=%.3f not recovering in %.1f s; re-planting '
                            'gait (recovery %d/%d).',
                            self.truth_pose[1],
                            self.segment_turn_push_ineffective_sec,
                            self.landing_slip_recoveries,
                            self.landing_slip_max_attempts)
                        self._record_trace()
                        return
                    rospy.logwarn('Segment turn y=%.3f above north guard '
                                  '%.2f; pushing back to %.2f before '
                                  'continuing.',
                                  self.truth_pose[1],
                                  self.segment_turn_turn_y_guard_north,
                                  self.segment_turn_y_guard_recenter)
                    self.landing_yaw_stall_accum=0.0
                    self._publish_truth_world_command(0.0, push, 0.0)
                    self._record_trace()
                    return
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
            vy=self._segment_turn_side_vy(
                self.segment_turn_y_low-self.truth_pose[1])
        elif self.truth_pose[1] > self.segment_turn_y_high:
            vy=self._segment_turn_side_vy(
                self.segment_turn_y_high-self.truth_pose[1])
        wz=max(-self.entry_alignment_yaw_rate,
               min(self.entry_alignment_yaw_rate, .90*error))
        # R46(2026-08-26 定稿):去掉 minimum_yaw_rate 下限强制。batch9
        # RUN1 实证:转体完成后 error≈0,下限强制在 error 过零处每 tick
        # 输出 ±0.24 交替 → 步态反复起步横跳,冻结 40s 超时。此阶段
        # error 已 ≤0.20(turn_done),纯比例 0.90*error 输出 ≤0.18 慢转
        # 即可收敛,不需要下限;下限只对转体(turn 阶段强转)有意义。
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
        # F2→F1 完成(全流程段 2 或 Phase B 单段):全程终态
        self._write_segment_json(self.segment_two_log_basename,
                                 'FIRST_FLOOR_RETURNED')
        self._record_trace(force=True)
        rospy.loginfo('Returned to first floor after %.1f s (drop=%.2f m).',
                      self._elapsed(), drop)
        # 主标志 JSON(含 wall 时间戳)
        self.save()
        try:
            trigger_wall = self.trigger_wall_time if self.trigger_wall_time \
                else (self.trace[0].get('wall_time') if self.trace else None)
            descent_elapsed = round(
                time.time() - trigger_wall, 3) if trigger_wall else None
            with open(os.path.join(self.out, 'first_floor_returned.json'),
                      'w') as stream:
                json.dump({
                    'phase': 'FIRST_FLOOR_RETURNED',
                    'wall_time': round(time.time(), 3),
                    'trigger_wall_time': round(trigger_wall, 3)
                    if trigger_wall else None,
                    'descent_elapsed_sec': descent_elapsed,
                    'elapsed_sec': round(self._elapsed(), 3),
                    'descent_total_drop': round(drop, 3),
                    'segment_one_phase': 'SECOND_FLOOR_DESCENT_REACHED',
                    'segment_one_elapsed_sec': self.segment_one_elapsed_sec,
                }, stream)
        except OSError as error:
            rospy.logerr('Cannot write first_floor_returned.json: %s', error)
        self.first_floor_state_pub.publish(String(data='FIRST_FLOOR_RETURNED'))
        self._terminal('FIRST_FLOOR_RETURNED', 'first_floor_returned')

    # ------------------------------------------------------------------ #
    # 保存
    # ------------------------------------------------------------------ #
    def _write_segment_json(self, basename, phase):
        try:
            with open(os.path.join(self.out, 'logs', basename), 'w') as stream:
                json.dump({'phase': phase,
                           'policy': self.policy,
                           'wall_time': round(time.time(), 3),
                           'trace': self.trace},
                          stream)
        except OSError as error:
            rospy.logerr('Cannot write %s: %s', basename, error)

    def save(self):
        try:
            with open(os.path.join(self.out, 'logs',
                                   self.transition_log_basename), 'w') as stream:
                json.dump({'phase': self.phase,
                           'policy': self.policy,
                           'wall_time': round(time.time(), 3),
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

    def _on_shutdown_save(self):
        """on_shutdown 钩子:只保存终态。

        中间 phase(FLIGHT_A 等)在外部 shutdown(launch teardown/信号)时
        被 save 写出,会被 wrapper 读成终态误判失败(batch17 run3 实证:
        读到 STAIR_DESCENT_FLIGHT_A 判败,实际 FIRST_FLOOR_RETURNED)。
        终态均已由 _terminal / 成功路径 save 过,此处仅兜底重复写。
        """
        if self.phase in ('FIRST_FLOOR_RETURNED',
                          'SECOND_FLOOR_DESCENT_REACHED') or \
                self.phase.endswith(('_TIMEOUT', '_FAILED', '_DETECTED',
                                     '_LOST', '_ABORT', '_NOT_REACHED')):
            self.save()


def main():
    rospy.init_node('stair_descent_manager')
    node=StairDescent()
    rospy.spin()


if __name__ == '__main__':
    main()
