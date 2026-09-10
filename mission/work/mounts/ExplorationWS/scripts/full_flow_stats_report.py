#!/usr/bin/env python3
"""全流程探索测试统计:各环节耗时(仿真时间)+ 房间 + 危险源检测。

用法:
    python3 scripts/full_flow_stats_report.py <run_dir_1> [<run_dir_2> ...]

对每组运行输出:
  1) 各环节耗时:墙钟 + 仿真时间(sim_time_calibration 校准流换算),
     含上行(startup/F1 探索/引导/爬梯/F2/F3)与下行各子阶段
     (ENTRY_GUIDE/PRE_ALIGN/POLICY_LOADING/WARMUP/FLIGHT_B/TURN/
      FLIGHT_A/SEGMENT_TURN/STAND/EAST_ALIGN);
  2) 房间探索:recognized/exited/target/complete + F2/F3 handoff 房间数;
  3) 危险源检测:真值源数 / 确认检测数(逐 id)+ 检测率。
纯标准库 + statistics_timing_report / sim_time_calibration,无 ROS 依赖。
"""

import json
import math
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..",
    "src", "simenv_exploration", "scripts"))

import statistics_timing_report as strt
import sim_time_calibration as stc

SEG_LABEL = {
    "f1_explore_sec": "F1 房间探索",
    "stair_guide_sec": "楼梯引导(F1→F2)",
    "stair_ascent_total_sec": "爬梯(F1→F2)",
    "f2_corridor_guide_sec": "F2 走廊引导",
    "f2_explore_sec": "F2 房间探索",
    "f2_to_f3_stair_sec": "爬梯(F2→F3)",
    "f3_explore_sec": "F3 房间探索",
    "descent_segment1_sec": "下行段1(F3→F2)",
    "descent_segment2_sec": "下行段2(F2→F1)",
    "descent_total_sec": "下行总(返程)",
    "total_sec": "全流程总",
}

# trace 时间线(段 0 = F3→F2,段 1 = F2→F1):
#   段0: ENTRY_GUIDE→PRE_ALIGN→(POLICY_LOADING/WARMUP)→STAND→
#        FLIGHT_B→TURN→FLIGHT_A
#   段1: SEGMENT_TURN→STAND→EAST_ALIGN→FLIGHT_B→TURN→FLIGHT_A→
#        FIRST_FLOOR_RETURNED
DESCENT_PHASES = [
    "STAIR_DESCENT_ENTRY_GUIDE", "STAIR_DESCENT_PRE_ALIGN",
    "STAIR_DESCENT_POLICY_LOADING", "STAIR_DESCENT_POLICY_WARMUP",
    "STAIR_DESCENT_FLIGHT_B", "STAIR_DESCENT_TURN",
    "STAIR_DESCENT_FLIGHT_A", "STAIR_DESCENT_SEGMENT_TURN",
    "STAIR_DESCENT_STAND", "STAIR_DESCENT_EAST_ALIGN",
    "STAIR_DESCENT_FIRST_FLOOR_RETURNED",
]

SEG0_LABEL = "下行段1(F3→F2)"
SEG1_LABEL = "下行段2(F2→F1)"

PHASE_SHORT = {
    "STAIR_DESCENT_ENTRY_GUIDE": "ENTRY_GUIDE",
    "STAIR_DESCENT_PRE_ALIGN": "PRE_ALIGN",
    "STAIR_DESCENT_POLICY_LOADING": "POLICY_LOAD",
    "STAIR_DESCENT_POLICY_WARMUP": "WARMUP",
    "STAIR_DESCENT_FLIGHT_B": "FLIGHT_B",
    "STAIR_DESCENT_TURN": "TURN",
    "STAIR_DESCENT_FLIGHT_A": "FLIGHT_A",
    "STAIR_DESCENT_SEGMENT_TURN": "SEGMENT_TURN",
    "STAIR_DESCENT_STAND": "STAND",
    "STAIR_DESCENT_EAST_ALIGN": "EAST_ALIGN",
}


def load(path):
    return strt.load_json(path)


def descent_subphases(run_dir, stream, run_rtf):
    """下行各子阶段墙钟/仿真耗时;返回 {segment: [(phase, wall_sec, sim_sec)]}。
    段1(F3→F2)与段2(F2→F1)的 FLIGHT_B/TURN/FLIGHT_A 同名,必须按 trace 的
    segment 字段分组;子阶段端点直接在校准流插值(wall→ros)。"""
    d = load(os.path.join(run_dir, "logs", "stair_descent.json"))
    if not d:
        return None
    trace = d.get("trace") or []
    bounds = {}  # (segment, phase) -> (first_wall, last_wall)
    for t in trace:
        w = strt.wall_seconds(t.get("wall_time"))
        if w is None:
            continue
        key = (t.get("segment", 0), t.get("phase"))
        if key in bounds:
            if w < bounds[key][0]:
                bounds[key] = (w, bounds[key][1])
            elif w > bounds[key][1]:
                bounds[key] = (bounds[key][0], w)
        else:
            bounds[key] = (w, w)
    out = {}
    for (seg, ph), (w0, w1) in sorted(bounds.items()):
        if w1 <= w0:
            continue
        r0 = stc.interp_ros(stream, w0) if stream else None
        r1 = stc.interp_ros(stream, w1) if stream else None
        if r0 is not None and r1 is not None and r1 >= r0:
            sim = r1 - r0
        elif run_rtf:
            sim = (w1 - w0) * run_rtf
        else:
            sim = None
        out.setdefault(seg, []).append(
            (PHASE_SHORT.get(ph, ph), w1 - w0, sim))
    return out


def hazard_stats(run_dir, xy_match=0.85, z_tol=1.0):
    """危险源:真值源总数 + 确认检测(按位置匹配,id 字段格式不一致)。

    真值 id 是数字(danger_truth.json),confirmed 的 id 是 "hazard-XX"
    字符串,无法直接比;真值 position 是 [x,y,z] 列表,confirmed 是
    list 或 {x,y,z} dict。confirmed 的 z = 真值 z + ~0.4(球心高于底面,
    实测 z 偏差 0.39-0.48 系统化)→ z 容差 1.0 作楼层过滤。xy 容差
    0.85 = 官方评估(visualize_baseline_results.py
    plot_danger_detection_evaluation tolerance=.85)口径;官方还会把
    FAST-LIO map 坐标变换到 world(偏移 0.3-0.6m),本函数用 0.85 容差
    覆盖该偏移,近似官方 true_positive_count。"""
    truth = load(os.path.join(run_dir, "danger_truth.json"))
    conf = load(os.path.join(run_dir, "confirmed_hazards.json"))
    if conf is None:
        return None
    truth_srcs = (truth or {}).get("danger_sources") or []
    truth_ids = [s.get("id") for s in truth_srcs]
    confirmed = conf.get("confirmed_hazards") or []
    det_ids = set()
    for h in confirmed:
        p = h.get("position")
        if isinstance(p, dict):
            pc = (p.get("x"), p.get("y"), p.get("z"))
        elif isinstance(p, (list, tuple)) and len(p) >= 3:
            pc = tuple(p[:3])
        else:
            continue
        if None in pc or not all(isinstance(v, (int, float)) for v in pc):
            continue
        best = None
        for s in truth_srcs:
            pt = s.get("position")
            if not pt:
                continue
            xy = math.sqrt((pc[0] - pt[0]) ** 2 + (pc[1] - pt[1]) ** 2)
            if xy <= xy_match and abs(pc[2] - pt[2]) <= z_tol:
                best = s.get("id")
                break
        if best is not None:
            det_ids.add(best)
    return {
        "n_truth": len(truth_ids),
        "n_det": len(det_ids),
        "det_ids": sorted(str(i) for i in det_ids),
        "truth_ids": sorted(str(i) for i in truth_ids),
        "n_confirmed": len(confirmed),
        "detected_danger_sources": conf.get("detected_danger_sources"),
    }


def room_stats(run_dir):
    bs = load(os.path.join(run_dir, "baseline_summary.json")) or {}
    h2 = load(os.path.join(run_dir, "second_floor_handoff.json")) or {}
    h3 = load(os.path.join(run_dir, "third_floor_handoff.json")) or {}
    return {
        "f1_recognized": bs.get("recognized_room_count"),
        "f1_exited": bs.get("exited_room_count"),
        "f1_target": bs.get("room_target_count"),
        "f1_complete": bs.get("complete_room_count"),
        "f1_success_goals": bs.get("goal_execution_success_count"),
        "f1_failed_goals": bs.get("goal_execution_failure_count"),
        "f1_frontier_terminations": bs.get("frontier_termination_decision_count"),
        "f2_recognized": h2.get("recognized_room_count"),
        "f2_exited": h2.get("exited_room_count"),
        "f3_recognized": h3.get("recognized_room_count"),
        "f3_exited": h3.get("exited_room_count"),
    }


def fmt(v, digits=1):
    if v is None:
        return "-"
    if isinstance(v, (list, tuple)):
        return ", ".join(str(x) for x in v)
    return "%.*f" % (digits, v)


def report_run(run_dir):
    name = os.path.basename(run_dir.rstrip("/"))
    metrics, notes = strt.analyze(run_dir)
    cal = stc.convert_run(run_dir, metrics)
    stream = stc.load_stream(run_dir, cal["calibration_source"] or "trajectory")
    run_rtf = cal.get("run_rtf")
    print("### %s  (RTF=%s, source=%s)" % (
        name, fmt(run_rtf, 4), cal.get("calibration_source")))
    # 1) 各环节耗时
    print("| 环节 | 墙钟(s) | 仿真(s) | 说明 |")
    print("|---|---|---|---|")
    for k in stc.SEG_KEYS:
        if k not in metrics and k not in cal["sim"]:
            continue
        label = SEG_LABEL.get(k, k)
        wall = metrics.get(k)
        sim = cal["sim"].get(k)
        rtf = (cal["rtfs"].get(k) or {}).get("rtf")
        note = "RTF=%.3f" % rtf if rtf else "-"
        print("| %s | %s | %s | %s |" % (label, fmt(wall), fmt(sim), note))
    if cal.get("sim_residual_sec") is not None:
        print("| *残差* | %s | %s | (总 − 各段和) |" % (
            fmt(metrics.get("residual_sec")), fmt(cal["sim_residual_sec"])))
    # 2) 下行子阶段(按段分组;seg 0 = F3→F2,seg 1 = F2→F1)
    subs = descent_subphases(run_dir, stream, run_rtf)
    if subs:
        for seg in sorted(subs):
            print("\n%s 子阶段(墙钟 → 仿真):"
                  % (SEG1_LABEL if seg else SEG0_LABEL))
            print("| 阶段 | 墙钟(s) | 仿真(s) |")
            print("|---|---|---|")
            for ph, wall, sim in subs[seg]:
                print("| %s | %s | %s |" % (ph, fmt(wall), fmt(sim)))
    # 3) 房间
    rs = room_stats(run_dir)
    if any(v is not None for v in rs.values()):
        print("\n房间探索:")
        print("- F1: 识别 %s / 走出 %s / 目标 %s / 完成 %s | 目标成功 %s 失败 %s | 前沿终止 %s"
              % (fmt(rs["f1_recognized"], 0), fmt(rs["f1_exited"], 0),
                 fmt(rs["f1_target"], 0), fmt(rs["f1_complete"], 0),
                 fmt(rs["f1_success_goals"], 0), fmt(rs["f1_failed_goals"], 0),
                 fmt(rs["f1_frontier_terminations"], 0)))
        print("- F2: 识别 %s / 走出 %s | F3: 识别 %s / 走出 %s"
              % (fmt(rs["f2_recognized"], 0), fmt(rs["f2_exited"], 0),
                 fmt(rs["f3_recognized"], 0), fmt(rs["f3_exited"], 0)))
    # 4) 危险源
    hs = hazard_stats(run_dir)
    if hs:
        rate = (hs["n_det"] / hs["n_truth"] if hs["n_truth"] else None)
        print("\n危险源检测: 位置匹配 %s / 真值 %s (%.0f%%) | 确认报告 %s 条 | 字段值 %s"
              % (fmt(hs["n_det"], 0), fmt(hs["n_truth"], 0),
                 (rate or 0) * 100 if rate is not None else 0,
                 fmt(hs["n_confirmed"], 0),
                 fmt(hs["detected_danger_sources"], 0)))
        print("- 匹配 id: %s" % (", ".join(hs["det_ids"]) if hs["det_ids"] else "(无)"))
        if hs["truth_ids"]:
            missing = set(hs["truth_ids"]) - set(hs["det_ids"])
            if missing:
                print("- 漏检 id: %s" % ", ".join(sorted(missing)))
    for n in notes:
        print("note: %s" % n)
    print()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    for run_dir in sys.argv[1:]:
        report_run(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
