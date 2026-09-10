#!/usr/bin/env python3
"""全流程测试墙钟 → 仿真时间换算模块。

背景: 全栈 use_sim_time=true 下 rospy/ros::Time 即仿真时钟,但阶段 trace
日志的时间戳全是墙上时钟(time.monotonic / time.time)。校准源
trajectory_timeseries.jsonl(5Hz) 与 map_growth_timeseries.jsonl(0.5Hz)
每条记录同时含 ros_time(仿真) 与 wall_time(墙钟),据此把墙钟时刻映射到
仿真时刻,仿真耗时 = ros(段末) − ros(段首)。

分段窗口:
  - 锚定段(F1/F2/F3 探索、下行、总时长): 段首末都是已知 wall_time,直接
    在校准流上线性插值求 ros 差 —— 精确等价 Δros,对仿真暂停天然正确;
  - trace 段(楼梯引导、爬梯): trace 的 t 是 monotonic 墙钟累计,跨度即墙钟
    时长,锚定 exp2/exp3_start(探索开始 wall_time,≈ trace 末尾)反推窗口
    [exp_start − 跨度, exp_start],同样端点插值换算。

用法:
    python3 scripts/sim_time_calibration.py <run_dir> [--source trajectory|map_growth]
或作为模块: from sim_time_calibration import convert_run
纯标准库 + 复用 statistics_timing_report 的纯函数,无 ROS 依赖。
"""

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..",
    "src", "simenv_exploration", "scripts"))

import statistics_timing_report as strt

# 每段最小样本数(按校准源;trajectory 5Hz≈2s,map_growth 0.5Hz≈6s)
MIN_SAMPLES = {"trajectory": 10, "map_growth": 3}
# 停顿对判定: 相邻样本 墙钟跨度>2s 而 仿真跨度<0.1s => 仿真暂停(墙钟照走)
PAUSE_WALL_MAX = 2.0
PAUSE_ROS_MIN = 0.1
# ros_time 回退容忍(>5ms 视为异常行)
ROS_REGRESS_TOL = 0.005
# 迭代 MAD 剔除
MAD_K = 3.0
MAX_MAD_ROUNDS = 2
# RTF 质量阈值
RTF_LOW, RTF_HIGH = 0.5, 10.0
GLOBAL_DEV_FRAC = 0.25
# 仿真残差 note 阈值
SIM_RESIDUAL_NOTE_S = 25.0
# 插值边界外容忍(秒): wall 超出校准流边界该距离内仍用端点值
INTERP_EDGE_TOL = 0.5

SEG_KEYS = ("f1_explore_sec", "stair_guide_sec", "stair_ascent_total_sec",
            "f2_corridor_guide_sec", "f2_explore_sec", "f2_to_f3_stair_sec",
            "f3_explore_sec", "descent_segment1_sec", "descent_segment2_sec",
            "descent_total_sec", "total_sec")


def load_stream(run_dir, source):
    """读校准流,返回按 wall 排序去重、清理后的 [(wall, ros), ...] 或 None。"""
    fname = {"trajectory": "trajectory_timeseries.jsonl",
             "map_growth": "map_growth_timeseries.jsonl"}[source]
    path = os.path.join(run_dir, fname)
    rows = []
    try:
        with open(path, encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                w, r = d.get("wall_time"), d.get("ros_time")
                if w is None or r is None:
                    continue
                if source == "trajectory" and d.get("clock_phase") != "MISSION":
                    continue
                rows.append((float(w), float(r)))
    except OSError:
        return None
    if not rows:
        return None
    rows.sort()
    out = []
    for w, r in rows:
        if out and abs(w - out[-1][0]) < 1e-9:
            out[-1] = (w, r)  # 同 wall 保留末条
        else:
            out.append((w, r))
    # ros_time 回退清理(剔除回退 >5ms 的行)
    clean = []
    for w, r in out:
        if clean and r < clean[-1][1] - ROS_REGRESS_TOL:
            continue
        clean.append((w, r))
    return clean or None


def interp_ros(stream, wall):
    """wall 时刻的 ros 线性插值;超出校准流边界 INTERP_EDGE_TOL 内用端点值,否则 None。"""
    if not stream:
        return None
    if wall <= stream[0][0]:
        return stream[0][1] if wall >= stream[0][0] - INTERP_EDGE_TOL else None
    if wall >= stream[-1][0]:
        return stream[-1][1] if wall <= stream[-1][0] + INTERP_EDGE_TOL else None
    lo, hi = 0, len(stream) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if stream[mid][0] <= wall:
            lo = mid
        else:
            hi = mid
    w0, r0 = stream[lo]
    w1, r1 = stream[hi]
    if w1 == w0:
        return r0
    return r0 + (r1 - r0) * (wall - w0) / (w1 - w0)


def drop_pause_pairs(points):
    """剔除停顿对: 相邻 Δwall>PAUSE_WALL_MAX 且 Δros<PAUSE_ROS_MIN 的两个点。"""
    if len(points) < 3:
        return points
    keep = [True] * len(points)
    for i in range(1, len(points)):
        dw = points[i][0] - points[i - 1][0]
        dr = points[i][1] - points[i - 1][1]
        if dw > PAUSE_WALL_MAX and dr < PAUSE_ROS_MIN:
            keep[i] = keep[i - 1] = False
    return [p for p, k in zip(points, keep) if k]


def _ols_slope(ws, rs):
    n = len(ws)
    if n < 2:
        return None
    mw = sum(ws) / n
    mr = sum(rs) / n
    cov = sum((w - mw) * (r - mr) for w, r in zip(ws, rs))
    var = sum((w - mw) ** 2 for w in ws)
    if var <= 0:
        return None
    return cov / var


def robust_rtf(points, min_samples):
    """窗口内稳健 OLS 斜率(仅作质量参考): 停顿剔除 + 迭代 MAD,返回 dict 或 None。"""
    pts = drop_pause_pairs(points)
    if len(pts) < 2:
        return None
    ws = [p[0] for p in pts]
    rs = [p[1] for p in pts]
    slope = _ols_slope(ws, rs)
    if slope is None or slope <= 0:
        return None
    n_start = len(ws)
    for _ in range(MAX_MAD_ROUNDS):
        pred = [slope * w for w in ws]
        resid = [r - p for r, p in zip(rs, pred)]
        med = statistics.median(resid)
        mad = statistics.median(abs(x - med) for x in resid)
        if mad <= 1e-9:
            break
        keep = [abs(x - med) <= MAD_K * mad for x in resid]
        if sum(keep) == len(keep):
            break
        ws = [w for w, k in zip(ws, keep) if k]
        rs = [r for r, k in zip(rs, keep) if k]
        slope = _ols_slope(ws, rs)
        if slope is None or slope <= 0:
            return None
    if len(ws) < min_samples:
        return None
    mr = sum(rs) / len(rs)
    ss_tot = sum((r - mr) ** 2 for r in rs)
    pred = [slope * w for w in ws]
    ss_res = sum((r - p) ** 2 for r, p in zip(rs, pred))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    return {"n": len(ws), "n_rejected": n_start - len(ws), "r2": r2}


def global_rtf(stream):
    """整轮 RTF = 校准流首末斜率(与 stair_sim_time.rtf_of 口径一致)。"""
    if not stream or len(stream) < 3:
        return None
    dr = stream[-1][1] - stream[0][1]
    dw = stream[-1][0] - stream[0][0]
    return dr / dw if dw > 0 else None


def segment_windows(run_dir, metrics):
    """各段墙钟窗口 [w0, w1];无法确定的段不出现在结果里。"""
    load_json = strt.load_json
    gh = load_json(os.path.join(run_dir, "goal_execution_history.json"))
    f1_start, _f2_ctx, f1_end = strt.goal_boundary_walls(gh)
    h2 = load_json(os.path.join(run_dir, "second_floor_handoff.json"))
    h3 = load_json(os.path.join(run_dir, "third_floor_handoff.json"))
    returned = load_json(os.path.join(run_dir, "first_floor_returned.json"))
    seg1 = load_json(os.path.join(run_dir, "logs",
                                  "third_to_second_floor_stair_transition.json"))
    seg2 = load_json(os.path.join(run_dir, "logs",
                                  "second_to_first_floor_stair_transition.json"))
    descent = load_json(os.path.join(run_dir, "logs", "stair_descent.json"))

    exp2 = strt.wall_seconds((h2 or {}).get("exploration_started_wall_time"))
    h2_done = strt.wall_seconds((h2 or {}).get("wall_time"))
    exp3 = strt.wall_seconds((h3 or {}).get("exploration_started_wall_time"))
    h3_done = strt.wall_seconds((h3 or {}).get("wall_time"))

    w = {}
    if f1_start is not None and f1_end is not None:
        w["f1_explore_sec"] = (f1_start, f1_end)
    # trace 类段: 指标即墙钟跨度,锚定 exp2/exp3_start 反推窗口
    climb12 = metrics.get("stair_ascent_total_sec")
    guide12 = metrics.get("stair_guide_sec")
    if exp2 is not None and climb12 is not None:
        w["stair_ascent_total_sec"] = (exp2 - climb12, exp2)
        if guide12 is not None:
            w["stair_guide_sec"] = (exp2 - climb12 - guide12, exp2 - climb12)
    guide2 = metrics.get("f2_corridor_guide_sec")
    if exp2 is not None and guide2 is not None:
        w["f2_corridor_guide_sec"] = (exp2, exp2 + guide2)
    if exp2 is not None and h2_done is not None:
        w["f2_explore_sec"] = (exp2, h2_done)
    climb23 = metrics.get("f2_to_f3_stair_sec")
    if exp3 is not None and climb23 is not None:
        w["f2_to_f3_stair_sec"] = (exp3 - climb23, exp3)
    if exp3 is not None and h3_done is not None:
        w["f3_explore_sec"] = (exp3, h3_done)

    trigger = strt.wall_seconds((returned or {}).get("trigger_wall_time"))
    if trigger is None:
        tr0 = ((descent or {}).get("trace") or [])
        if tr0:
            trigger = strt.wall_seconds(tr0[0].get("wall_time"))
    seg1_wall = strt.wall_seconds((seg1 or {}).get("wall_time"))
    seg2_wall = strt.wall_seconds((seg2 or {}).get("wall_time"))
    returned_wall = strt.wall_seconds((returned or {}).get("wall_time"))
    if trigger is not None and seg1_wall is not None:
        w["descent_segment1_sec"] = (trigger, seg1_wall)
    if seg1_wall is not None and seg2_wall is not None:
        w["descent_segment2_sec"] = (seg1_wall, seg2_wall)
    if trigger is not None and (returned_wall or seg2_wall) is not None:
        w["descent_total_sec"] = (trigger, returned_wall or seg2_wall)
    if f1_start is not None:
        end = returned_wall or h2_done
        if end is not None:
            w["total_sec"] = (f1_start, end)
    return w


def convert_run(run_dir, metrics, source="trajectory"):
    """换算一轮。返回:
    {sim: {seg: 仿真秒}, sim_residual_sec, rtfs: {seg: {rtf,n,n_rejected,r2,source}},
     run_rtf, calibration_source, notes:[...]}
    """
    out = {"sim": {}, "sim_residual_sec": None, "rtfs": {},
           "run_rtf": None, "calibration_source": "none", "notes": []}
    stream = load_stream(run_dir, source)
    used = source
    if stream is None and source == "trajectory":
        used = "map_growth"
        stream = load_stream(run_dir, "map_growth")
    if stream is None:
        out["notes"].append("RTF 不可用(无校准数据),耗时保留墙钟")
        return out
    out["calibration_source"] = used
    run_rtf = global_rtf(stream)
    out["run_rtf"] = run_rtf
    if run_rtf is None:
        out["notes"].append("RTF 不可用(校准数据不足),耗时保留墙钟")
        return out
    if not (RTF_LOW <= run_rtf <= RTF_HIGH):
        out["notes"].append("可疑RTF(%.3f)" % run_rtf)
    min_n = MIN_SAMPLES[used]
    windows = segment_windows(run_dir, metrics)
    for key, (w0, w1) in windows.items():
        mval = metrics.get(key)
        if mval is None:
            continue
        span = w1 - w0
        if span <= 0:
            continue
        pts = [p for p in stream if w0 <= p[0] <= w1]
        r0 = interp_ros(stream, w0)
        r1 = interp_ros(stream, w1)
        note = None
        if r0 is not None and r1 is not None and r1 >= r0:
            sim = r1 - r0
            rtf = sim / span
        else:
            sim = mval * run_rtf
            rtf = run_rtf
            note = "RTF回退全局"
        quality = robust_rtf(pts, min_n) or {"n": len(pts),
                                             "n_rejected": None, "r2": None}
        out["sim"][key] = sim
        out["rtfs"][key] = dict(quality, rtf=rtf, source=used)
        if note:
            out["notes"].append("%s:%s" % (key, note))
        if not (RTF_LOW <= rtf <= RTF_HIGH):
            out["notes"].append("%s:可疑RTF(%.3f)" % (key, rtf))
        elif run_rtf and abs(rtf - run_rtf) / run_rtf > GLOBAL_DEV_FRAC:
            out["notes"].append("%s:段RTF偏离全局(%.3f vs %.3f)"
                                % (key, rtf, run_rtf))

    # 仿真残差 = 仿真总时长 − 各段仿真之和(口径同 strt.analyze 的残差)
    base = ("f1_explore_sec", "stair_guide_sec", "stair_ascent_total_sec",
            "f2_corridor_guide_sec", "f2_explore_sec")
    total = out["sim"].get("total_sec")
    if total is not None and all(out["sim"].get(k) is not None for k in base):
        s = sum(out["sim"][k] for k in base)
        for k in ("f2_to_f3_stair_sec", "f3_explore_sec", "descent_total_sec"):
            if out["sim"].get(k) is not None:
                s += out["sim"][k]
        resid = total - s
        out["sim_residual_sec"] = resid
        if resid > SIM_RESIDUAL_NOTE_S:
            out["notes"].append("仿真残差>%gs(%.1f)" % (SIM_RESIDUAL_NOTE_S, resid))
    return out


def _fmt(v, digits=1):
    return "-" if v is None else ("%.*f" % (digits, v))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="一轮结果目录(fuel_rooffix_fullN)")
    parser.add_argument("--source", choices=("trajectory", "map_growth"),
                        default="trajectory")
    args = parser.parse_args(argv)

    metrics, strt_notes = strt.analyze(args.run_dir)
    cal = convert_run(args.run_dir, metrics, args.source)
    print("run=%s  source=%s  run_rtf=%s  wall_total=%s  sim_total=%s" % (
        os.path.basename(args.run_dir), cal["calibration_source"],
        _fmt(cal["run_rtf"], 4), _fmt(metrics.get("total_sec")),
        _fmt(cal["sim"].get("total_sec"))))
    for k in SEG_KEYS:
        ftf = (cal["rtfs"].get(k) or {}).get("rtf")
        print("  %-24s wall=%8s sim=%8s rtf=%s n=%s" % (
            k, _fmt(metrics.get(k)), _fmt(cal["sim"].get(k)),
            _fmt(ftf, 4), (cal["rtfs"].get(k) or {}).get("n", "-")))
    if cal["sim_residual_sec"] is not None:
        print("  sim_residual=%s (wall_residual=%s)"
              % (_fmt(cal["sim_residual_sec"]), _fmt(metrics.get("residual_sec"))))
    for n in cal["notes"]:
        print("note: %s" % n)
    for n in strt_notes:
        print("note(strt): %s" % n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
