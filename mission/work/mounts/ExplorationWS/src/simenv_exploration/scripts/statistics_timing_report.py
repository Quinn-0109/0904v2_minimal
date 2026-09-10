#!/usr/bin/env python3
"""统计多组 F1/F2 完整房间探索运行的分阶段耗时。

用法:
    python3 statistics_timing_report.py <output_dir_1> [<output_dir_2> ...]

对每组运行读取:
  <dir>/goal_execution_history.json      F1 起点(首个 goal_received)、F1/F2 边界
                                         (second_floor_pose_bounds_activated)
  <dir>/logs/stair_transition.json       trace[].t/.phase: 楼梯引导与爬梯分段
  <dir>/second_floor_gt_corridor_guide.json  F2 走廊引导三段
  <dir>/second_floor_handoff.json        F2 探索时长(EXPLORATION_COMPLETE)

输出 Markdown 表格到 stdout(纯标准库,无 ROS 依赖)。
"""

import json
import math
import os
import sys


def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, ValueError):
        return None


def wall_seconds(value):
    """解析 wall_time 字段,返回数值秒或 None。"""
    if value is None:
        return None
    return float(value)


def goal_boundary_walls(goal_history):
    """返回 (f1_start_wall, f2_context_wall, f1_end_wall)。

    f1_start_wall  = 首个 goal_received 的 wall_time
    f2_context_wall = 首个 second_floor_pose_bounds_activated 的 wall_time
    f1_end_wall     = f2_context_wall 之前最后一个 goal 事件的 wall_time
    """
    events = (goal_history or {}).get("events") or []
    f1_start = None
    f2_context = None
    last_goal_before_f2 = None
    for event in events:
        stamp = wall_seconds(event.get("wall_time"))
        name = event.get("event")
        if name == "goal_received" and f1_start is None and stamp is not None:
            f1_start = stamp
        if name == "second_floor_pose_bounds_activated":
            if f2_context is None and stamp is not None:
                f2_context = stamp
            if f2_context is not None and stamp is not None:
                last_goal_before_f2 = stamp
    if f2_context is not None:
        # The first pass above also wrote the f2_context wall time into
        # last_goal_before_f2; recompute from a clean slate over the F1 goal
        # events only.
        last_goal_before_f2 = None
        for event in events:
            stamp = wall_seconds(event.get("wall_time"))
            if stamp is None:
                continue
            if event.get("event") == "goal_received" and stamp < f2_context:
                if last_goal_before_f2 is None or stamp > last_goal_before_f2:
                    last_goal_before_f2 = stamp
    return f1_start, f2_context, last_goal_before_f2


def stair_phases(stair_json):
    """返回 (phase 名 -> 该 phase 的 (首个全局 t, 末个全局 t))。

    trace 的时间轴在爬梯开始时重置(楼梯管理器重新计时),
    因此把回退处断开成多个轴段,并累计为全局时间。
    """
    trace = (stair_json or {}).get("trace") or []
    if not trace:
        return {}
    # 找出轴段边界(时间回退处)
    segments = []
    current = []
    for item in trace:
        if current and item["t"] < current[-1]["t"] - 0.05:
            segments.append(current)
            current = []
        current.append(item)
    if current:
        segments.append(current)
    # 计算每段的全局偏移
    phases = {}
    offset = 0.0
    for segment in segments:
        start_t = segment[0]["t"]
        end_t = segment[-1]["t"]
        for item in segment:
            global_t = offset + (item["t"] - start_t)
            phase = item.get("phase")
            entry = item.get("entry_stage")
            bucket = phases.setdefault(phase, {"first": None, "last": None,
                                               "first_entry": None})
            if bucket["first"] is None:
                bucket["first"] = global_t
            bucket["last"] = global_t
            if bucket["first_entry"] is None and entry:
                bucket["first_entry"] = entry
        offset += (end_t - start_t)
    return phases


def guide_stages(guide_json):
    """返回 (stage -> (first_t, last_t)) 与总时长。"""
    trace = (guide_json or {}).get("trace") or []
    if not trace:
        return {}, None
    stages = {}
    for item in trace:
        stage = item.get("stage", "?")
        stamp = item.get("t")
        if stamp is None:
            continue
        bucket = stages.setdefault(stage, [None, None])
        if bucket[0] is None or stamp < bucket[0]:
            bucket[0] = stamp
        if bucket[1] is None or stamp > bucket[1]:
            bucket[1] = stamp
    total = max(v[1] for v in stages.values()) - \
        min(v[0] for v in stages.values())
    return stages, total


def trace_phase_bounds(trace):
    """trace 内每个 phase 的 (first_t, last_t),t 为轴内累计时间。"""
    phases = {}
    for item in trace:
        phase = item.get("phase")
        stamp = item.get("t")
        if phase is None or stamp is None:
            continue
        bucket = phases.setdefault(phase, [None, None])
        if bucket[0] is None or stamp < bucket[0]:
            bucket[0] = stamp
        if bucket[1] is None or stamp > bucket[1]:
            bucket[1] = stamp
    return phases


def descent_metrics(run_dir):
    """返程(F3→F2→F1)指标:各段墙钟时长 + 守卫计数 + 相位分段。

    时间锚点(均为 wall_time):
      下梯起点 = first_floor_returned.json.trigger_wall_time
                (经理 on_mission_trigger 记录;缺省回退 trace 首项)
      段 1 完成 = third_to_second_floor_stair_transition.json.wall_time
      段 2 完成 = second_to_first_floor_stair_transition.json.wall_time
      返程总   = first_floor_returned.json.descent_elapsed_sec(trigger 起算)
    """
    result = {}
    descent = load_json(os.path.join(run_dir, "logs", "stair_descent.json"))
    seg1 = load_json(os.path.join(run_dir, "logs",
                                  "third_to_second_floor_stair_transition.json"))
    seg2 = load_json(os.path.join(run_dir, "logs",
                                  "second_to_first_floor_stair_transition.json"))
    returned = load_json(os.path.join(run_dir, "first_floor_returned.json"))
    trace = (descent or {}).get("trace") or []
    if not descent and not seg1 and not seg2:
        return result
    start_wall = wall_seconds(trace[0].get("wall_time")) if trace else None
    seg1_wall = wall_seconds((seg1 or {}).get("wall_time"))
    seg2_wall = wall_seconds((seg2 or {}).get("wall_time"))
    # 下梯起点 = trigger 时刻(经理 on_mission_trigger 记录),
    # 回退到 trace 首项;生产链路中 manager 随探索一起启动,
    # trace[0] 会包含 F3 探索时间,故 trigger 锚点优先。
    trigger_wall = wall_seconds((returned or {}).get("trigger_wall_time"))
    if trigger_wall is None and start_wall is not None:
        trigger_wall = start_wall
    if trigger_wall is not None and seg1_wall is not None:
        result["descent_segment1_sec"] = seg1_wall - trigger_wall
    if seg1_wall is not None and seg2_wall is not None:
        result["descent_segment2_sec"] = seg2_wall - seg1_wall
    if (returned or {}).get("descent_elapsed_sec") is not None:
        result["descent_total_sec"] = returned["descent_elapsed_sec"]
    elif trigger_wall is not None and seg2_wall is not None:
        result["descent_total_sec"] = seg2_wall - trigger_wall
    result["descent_phase"] = (descent or {}).get("phase")
    result["descent_seg1_phase"] = (seg1 or {}).get("phase")
    result["descent_seg2_phase"] = (seg2 or {}).get("phase")
    result["descent_phases"] = trace_phase_bounds(trace)
    flight_b = [s for s in trace if s.get("phase") == "STAIR_DESCENT_FLIGHT_B"]
    if flight_b:
        result["descent_flight_b_stalls"] = max(
            (s.get("stall_recoveries") or 0) for s in flight_b)
        result["descent_flight_b_slips"] = max(
            (s.get("side_slip_attempts") or 0) for s in flight_b)
    if returned:
        result["descent_returned_wall"] = wall_seconds(returned.get("wall_time"))
        result["descent_returned_elapsed"] = returned.get("elapsed_sec")
    return result


def analyze(run_dir):
    """分析单组运行,返回 (指标 dict, 备注 list)。"""
    result = {}
    notes = []

    goal_history = load_json(os.path.join(run_dir, "goal_execution_history.json"))
    stair = load_json(os.path.join(run_dir, "logs", "stair_transition.json"))
    guide = load_json(os.path.join(run_dir,
                                   "second_floor_gt_corridor_guide.json"))
    handoff = load_json(os.path.join(run_dir, "second_floor_handoff.json"))

    f1_start, f2_ctx, f1_end = goal_boundary_walls(goal_history)
    if f1_start is not None and f1_end is not None:
        result["f1_explore_sec"] = f1_end - f1_start
        result["f1_start_wall"] = f1_start
    if f2_ctx is not None:
        result["f1_to_f2_handoff_wall"] = f2_ctx

    phases = stair_phases(stair)
    entry = phases.get("TRUTH_ENTRY_GUIDE")
    if entry and entry["first"] is not None:
        result["stair_guide_sec"] = entry["last"] - entry["first"]
        result["stair_entry_first_stage"] = entry.get("first_entry")
    pre_align = phases.get("STAIR_PRE_ASCENT_ALIGN")
    handoff_phase = (phases.get("SECOND_FLOOR_HANDOFF_COMPLETE") or
                     phases.get("SECOND_FLOOR_REACHED"))
    if pre_align and pre_align["first"] is not None and \
            handoff_phase and handoff_phase["last"] is not None:
        result["stair_ascent_total_sec"] = \
            handoff_phase["last"] - pre_align["first"]
    if stair:
        entry_guide = stair.get("entry_guide") or {}
        result["entry_stage_switches"] = entry_guide.get("stage_switches")
        result["entry_two_stage"] = entry_guide.get("two_stage_guide")
        result["entry_final_stage"] = entry_guide.get("stage")
        if entry_guide.get("two_stage_guide") is True:
            notes.append("two-stage")
        else:
            notes.append("three-stage")

    stages, guide_total = guide_stages(guide)
    if guide_total is not None:
        result["f2_corridor_guide_sec"] = guide_total
        result["f2_guide_stages"] = stages
    if guide:
        result["f2_guide_state"] = guide.get("state")

    if handoff:
        start = wall_seconds(handoff.get("exploration_started_wall_time"))
        done = wall_seconds(handoff.get("wall_time"))
        if start is not None and done is not None:
            result["f2_explore_sec"] = done - start
        result["f2_handoff_state"] = handoff.get("state")
        result["f2_termination_reason"] = handoff.get("termination_reason")
        if handoff.get("tour_mode"):
            notes.append("tour-mode")
        f1_start_wall = result.get("f1_start_wall")
        if f1_start_wall is not None and done is not None and \
                "descent_returned_wall" not in result:
            result["total_sec"] = done - f1_start_wall

    # ---- F2→F3 爬梯段 ----
    stair23 = load_json(os.path.join(run_dir, "logs",
                                     "second_to_third_floor_stair_transition.json"))
    if stair23:
        trace23 = stair23.get("trace") or []
        if trace23:
            bounds = trace_phase_bounds(trace23)
            pre = bounds.get("STAIR_PRE_ASCENT_ALIGN")
            handoff23 = (bounds.get("THIRD_FLOOR_HANDOFF_COMPLETE") or
                         bounds.get("THIRD_FLOOR_REACHED") or
                         bounds.get("THIRD_FLOOR_SETTLE"))
            if pre and pre[0] is not None and handoff23 and \
                    handoff23[1] is not None:
                result["f2_to_f3_stair_sec"] = handoff23[1] - pre[0]
        result["f2_to_f3_phase"] = stair23.get("phase")

    # ---- F3 探索段 ----
    handoff3 = load_json(os.path.join(run_dir, "third_floor_handoff.json"))
    if handoff3:
        start3 = wall_seconds(handoff3.get("exploration_started_wall_time"))
        done3 = wall_seconds(handoff3.get("wall_time"))
        if start3 is not None and done3 is not None:
            result["f3_explore_sec"] = done3 - start3
        result["f3_handoff_state"] = handoff3.get("state")

    # ---- 返程段(F3→F2→F1) ----
    result.update(descent_metrics(run_dir))
    if "descent_returned_wall" in result:
        f1_start_wall = result.get("f1_start_wall")
        done = result["descent_returned_wall"]
        if f1_start_wall is not None and done is not None:
            result["total_sec"] = done - f1_start_wall

    # 自检:总时长与各段之和的残差
    if "total_sec" in result and "f1_explore_sec" in result and \
            "stair_guide_sec" in result and "stair_ascent_total_sec" in result and \
            "f2_corridor_guide_sec" in result and "f2_explore_sec" in result:
        residual = result["total_sec"] - (
            result["f1_explore_sec"] + result["stair_guide_sec"] +
            result["stair_ascent_total_sec"] + result["f2_corridor_guide_sec"] +
            result["f2_explore_sec"])
        if "f2_to_f3_stair_sec" in result:
            residual -= result["f2_to_f3_stair_sec"]
        if "f3_explore_sec" in result:
            residual -= result["f3_explore_sec"]
        if "descent_total_sec" in result:
            residual -= result["descent_total_sec"]
        result["residual_sec"] = residual
        if residual > 30.0:
            notes.append("residual>30s")
    return result, notes


def fmt(value, digits=1):
    if value is None:
        return "-"
    if isinstance(value, float) and math.isnan(value):
        return "-"
    return "{:.{digits}f}".format(value, digits=digits)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    runs = []
    for path in sys.argv[1:]:
        if not os.path.isdir(path):
            print("跳过不存在目录: {}".format(path), file=sys.stderr)
            continue
        result, notes = analyze(path)
        runs.append((path, result, notes))
    if not runs:
        print("没有可分析的目录", file=sys.stderr)
        return 1

    header = ["组", "F1探索(s)", "楼梯引导(s)", "爬梯(s)", "F2走廊引导(s)",
              "F2探索(s)", "F2→F3爬梯(s)", "F3探索(s)", "下梯F3→F2(s)",
              "下梯F2→F1(s)", "返程总(s)", "总时长(s)", "残差(s)",
              "引导模式", "备注"]
    rows = []
    for path, result, notes in runs:
        guide_total = result.get("f2_corridor_guide_sec")
        guide_detail = ""
        if result.get("f2_guide_stages"):
            parts = []
            for stage, (first, last) in result["f2_guide_stages"].items():
                parts.append("{}:{:.1f}".format(stage.split("_corridor_")[-1]
                                                 if "_corridor_" in stage
                                                 else stage.split("_")[-1],
                                                 last - first))
            guide_detail = "[" + "+".join(parts) + "]"
        descent_note = ""
        if result.get("descent_flight_b_stalls") not in (None, 0):
            descent_note += "下梯stall={}".format(
                result["descent_flight_b_stalls"])
        rows.append([
            os.path.basename(path),
            fmt(result.get("f1_explore_sec")),
            fmt(result.get("stair_guide_sec")),
            fmt(result.get("stair_ascent_total_sec")),
            (fmt(guide_total) + guide_detail) if guide_total is not None else "-",
            fmt(result.get("f2_explore_sec")),
            fmt(result.get("f2_to_f3_stair_sec")),
            fmt(result.get("f3_explore_sec")),
            fmt(result.get("descent_segment1_sec")),
            fmt(result.get("descent_segment2_sec")),
            fmt(result.get("descent_total_sec")),
            fmt(result.get("total_sec")),
            fmt(result.get("residual_sec")),
            "两段" if result.get("entry_two_stage") else
            ("三段" if result.get("entry_two_stage") is False else "-"),
            ", ".join(notes) if notes else "",
        ])
    widths = [max(len(header[i]),
                  max((len(r[i]) for r in rows), default=0))
              for i in range(len(header))]
    def line(cells):
        return "| " + " | ".join(
            cell.ljust(widths[i]) for i, cell in enumerate(cells)) + " |"
    print(line(header))
    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for row in rows:
        print(line(row))

    # 引导模式与阶段切换检查
    print("\n检查:楼梯入口引导 stage_switches(两段式应为 1,三段式为 2)")
    for path, result, notes in runs:
        print("  {}: two_stage={} stage_switches={} final_stage={} "
              "首段={} 走廊引导state={}".format(
                  os.path.basename(path),
                  result.get("entry_two_stage"),
                  result.get("entry_stage_switches"),
                  result.get("entry_final_stage"),
                  result.get("stair_entry_first_stage"),
                  result.get("f2_guide_state")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
